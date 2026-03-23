from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

_MAX_LENGTH = 512
_CHUNK_OVERLAP_WORDS = 30

_BOUNDARY_PUNCT = set(".,;:!?\"'()[]{})")


@dataclass
class Entity:
    """A detected PII entity."""

    text: str
    label: str
    start_token: int
    end_token: int
    prefix: str = ""
    suffix: str = ""


class PiiModel:
    """HuggingFace token-classification wrapper for PII detection & anonymization."""

    def __init__(
        self,
        *,
        device: str | None = None,
        max_length: int = _MAX_LENGTH,
    ) -> None:
        self.model_id: str | None = None
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForTokenClassification | None = None
        self.id2label: dict[int, str] = {}

    def from_pretrained(self, model_id: str) -> PiiModel:
        """Load a HuggingFace token-classification model.

        Returns *self* so you can chain: ``PiiModel().from_pretrained("bardsai/...")``.
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()
        self.id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        return self

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call .from_pretrained(model_id) first."
            )

    def __repr__(self) -> str:
        return f"PiiModel(model_id={self.model_id!r}, device={self.device!r})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_pii(self, text: str) -> dict[str, str]:
        """Return ``{entity_text: LABEL, ...}`` for every PII span found."""
        self._ensure_loaded()
        entities = self._detect(text)
        return {e.text: e.label for e in entities}

    def anonymize(
        self,
        text: str,
        *,
        mode: Literal["simple", "ids"] = "simple",
    ) -> str | dict:
        """Replace PII spans with placeholders.

        * ``mode="simple"`` — returns string with ``[LABEL]`` placeholders.
        * ``mode="ids"``    — returns dict with indexed placeholders and mapping.
        """
        self._ensure_loaded()
        tokens = text.split()
        entities = self._detect(text)

        if mode == "ids":
            return self._anonymize_ids(tokens, entities)
        return self._anonymize_simple(tokens, entities)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_tags(self, tokens: list[str]) -> list[str]:
        """Predict BIO tags, automatically chunking texts that exceed max_length."""
        _tok_logger = logging.getLogger("transformers.tokenization_utils_base")
        prev_level = _tok_logger.level
        _tok_logger.setLevel(logging.ERROR)
        enc_full = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
        )
        _tok_logger.setLevel(prev_level)

        if len(enc_full["input_ids"]) <= self.max_length:
            return self._infer_chunk(tokens)

        subtoken_counts = [0] * len(tokens)
        for wid in enc_full.word_ids():
            if wid is not None:
                subtoken_counts[wid] += 1

        budget = self.max_length - 2  # [CLS] + [SEP]

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < len(tokens):
            cost = 0
            end = start
            while end < len(tokens) and cost + subtoken_counts[end] <= budget:
                cost += subtoken_counts[end]
                end += 1
            if end == start:
                end = start + 1
            chunks.append((start, end))
            if end >= len(tokens):
                break
            start = max(start + 1, end - _CHUNK_OVERLAP_WORDS)

        chunk_results: list[tuple[int, int, list[str]]] = []
        for cs, ce in chunks:
            tags = self._infer_chunk(tokens[cs:ce])
            chunk_results.append((cs, ce, tags))

        all_tags: list[str] = ["O"] * len(tokens)
        best_dist: list[float] = [float("inf")] * len(tokens)

        for cs, ce, tags in chunk_results:
            half = (ce - cs) / 2
            for local_i, tag in enumerate(tags):
                dist = abs(local_i - half)
                if dist < best_dist[cs + local_i]:
                    best_dist[cs + local_i] = dist
                    all_tags[cs + local_i] = tag

        return all_tags

    def _infer_chunk(self, tokens: list[str]) -> list[str]:
        """Run inference on a token list that fits within max_length."""
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

        word_ids = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        ).word_ids()

        out: list[str] = []
        prev_word_id = None
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                out.append(self.id2label[int(pred_ids[token_idx])])
            prev_word_id = word_id
        return out

    def _detect(self, text: str) -> list[Entity]:
        """Run inference and group BIO tags into Entity spans."""
        tokens = text.split()
        tags = self._predict_tags(tokens)

        entities: list[Entity] = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                label = tag[2:]
                start = i
                i += 1
                while i < len(tags) and tags[i] == f"I-{label}":
                    i += 1

                span_tokens = list(tokens[start:i])
                prefix, suffix = "", ""

                first = span_tokens[0]
                idx = 0
                while idx < len(first) and first[idx] in _BOUNDARY_PUNCT:
                    idx += 1
                if idx and idx < len(first):
                    prefix = first[:idx]
                    span_tokens[0] = first[idx:]

                last = span_tokens[-1]
                idx = len(last)
                while idx > 0 and last[idx - 1] in _BOUNDARY_PUNCT:
                    idx -= 1
                if idx < len(last) and idx > 0:
                    suffix = last[idx:]
                    span_tokens[-1] = last[:idx]

                entity_text = " ".join(span_tokens)
                entities.append(
                    Entity(entity_text, label, start, i, prefix, suffix)
                )
            else:
                i += 1
        return entities

    @staticmethod
    def _anonymize_simple(tokens: list[str], entities: list[Entity]) -> str:
        result = list(tokens)
        for ent in reversed(entities):
            result[ent.start_token : ent.end_token] = [
                f"{ent.prefix}[{ent.label}]{ent.suffix}"
            ]
        return " ".join(result)

    @staticmethod
    def _anonymize_ids(tokens: list[str], entities: list[Entity]) -> dict:
        counters: dict[str, int] = defaultdict(int)
        entity_keys: list[str] = []
        for ent in entities:
            counters[ent.label] += 1
            entity_keys.append(f"{ent.label}:{counters[ent.label]}")

        mapping: dict[str, str] = {}
        result = list(tokens)
        for ent, key in reversed(list(zip(entities, entity_keys))):
            mapping[key] = ent.text
            result[ent.start_token : ent.end_token] = [
                f"{ent.prefix}[{key}]{ent.suffix}"
            ]

        return {"anonymized_text": " ".join(result), "entities": mapping}
