from __future__ import annotations

import os
from typing import Protocol

from privacy_kit.core.types import Span

DEFAULT_MODEL_ID = "bardsai/eu-pii-anonimization-multilang"


class Detector(Protocol):
    def detect(self, text: str) -> list[Span]:
        ...


class _EncodingChunk:
    def __init__(self, ids, attention_mask, type_ids, offsets, word_ids) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids
        self.offsets = offsets
        self.word_ids = word_ids


class BardsAiOnnxDetector:
    """Local ONNX Runtime backend for bardsai/eu-pii-anonimization-multilang."""

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = 0.5,
        max_tokens: int = 512,
        stride: int = 128,
    ) -> None:
        if stride >= max_tokens:
            raise ValueError("stride must be smaller than max_tokens")
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.stride = stride

        try:
            import json

            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PII redaction model dependencies are missing. Install the package with `pip install privacy-kit`."
            ) from exc

        resolved_model_id = model_id or os.getenv("PII_MODEL_ID", DEFAULT_MODEL_ID)
        cache_dir = os.getenv("PII_MODEL_CACHE_DIR")

        try:
            tokenizer_path = hf_hub_download(resolved_model_id, "tokenizer.json", cache_dir=cache_dir)
            config_path = hf_hub_download(resolved_model_id, "config.json", cache_dir=cache_dir)
            model_path = hf_hub_download(resolved_model_id, "onnx/model_quantized.onnx", cache_dir=cache_dir)
        except Exception as exc:
            cache_hint = f" using cache dir {cache_dir!r}" if cache_dir else ""
            raise RuntimeError(
                f"Could not download or load model files for {resolved_model_id!r}{cache_hint}. "
                "Check network access, Hugging Face availability, or set PII_MODEL_CACHE_DIR to a writable directory."
            ) from exc

        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)

        self.id2label = {int(key): value for key, value in config["id2label"].items()}
        self._np = np
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_tokens, stride=stride)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            self._session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not start the local ONNX PII model from {model_path!r}. "
                "Check that onnxruntime supports this host and that the cached model file is not corrupted."
            ) from exc
        self._input_names = {model_input.name for model_input in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []

        spans: list[Span] = []
        for encoding in self._chunked_encodings(text):
            spans.extend(self._detect_encoding(encoding))
        return _dedupe_and_merge(spans)

    def _chunked_encodings(self, text: str):
        self._tokenizer.no_truncation()
        encoding = self._tokenizer.encode(text)
        self._tokenizer.enable_truncation(max_length=self.max_tokens, stride=self.stride)
        if len(encoding.ids) <= self.max_tokens:
            return [encoding]

        prefix_count = 0
        while prefix_count < len(encoding.ids) and encoding.word_ids[prefix_count] is None:
            prefix_count += 1

        suffix_count = 0
        while (
            suffix_count < len(encoding.ids) - prefix_count
            and encoding.word_ids[len(encoding.ids) - suffix_count - 1] is None
        ):
            suffix_count += 1

        body_start = prefix_count
        body_end = len(encoding.ids) - suffix_count
        body_capacity = self.max_tokens - prefix_count - suffix_count
        step = body_capacity - self.stride
        if body_capacity <= 0 or step <= 0:
            raise ValueError("stride must leave room for non-special tokens")

        chunks = []
        start = body_start
        while start < body_end:
            end = min(start + body_capacity, body_end)
            token_indexes = [*range(0, prefix_count), *range(start, end), *range(body_end, len(encoding.ids))]
            chunks.append(
                _EncodingChunk(
                    ids=[encoding.ids[index] for index in token_indexes],
                    attention_mask=[encoding.attention_mask[index] for index in token_indexes],
                    type_ids=[encoding.type_ids[index] for index in token_indexes],
                    offsets=[encoding.offsets[index] for index in token_indexes],
                    word_ids=[encoding.word_ids[index] for index in token_indexes],
                )
            )
            if end == body_end:
                break
            start += step
        return chunks

    def _detect_encoding(self, encoding) -> list[Span]:
        np = self._np
        inputs = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self._session.run([self._output_name], inputs)
        logits = outputs[0][0]
        probabilities = _softmax(logits)
        label_ids = probabilities.argmax(axis=-1)
        scores = probabilities.max(axis=-1)

        return self._bio_to_spans(
            label_ids=label_ids,
            scores=scores,
            offsets=encoding.offsets,
            word_ids=encoding.word_ids,
        )

    def _bio_to_spans(
        self,
        label_ids,
        scores,
        offsets: list[tuple[int, int]],
        word_ids: list[int | None],
    ) -> list[Span]:
        spans: list[Span] = []
        current_start: int | None = None
        current_end: int | None = None
        current_label: str | None = None
        current_scores: list[float] = []

        for label_id, score, offset, word_id in zip(label_ids, scores, offsets, word_ids, strict=True):
            start, end = offset
            if start == end or word_id is None:
                continue

            raw_label = self.id2label[int(label_id)]
            if raw_label == "O" or float(score) < self.threshold:
                if current_start is not None and current_label is not None and current_end is not None:
                    spans.append(Span(current_start, current_end, current_label, min(current_scores)))
                current_start = current_end = current_label = None
                current_scores = []
                continue

            prefix, entity = raw_label.split("-", 1)
            starts_new_span = (
                prefix == "B"
                or current_label != entity
                or current_start is None
                or (current_end is not None and start > current_end + 1)
            )
            if starts_new_span:
                if current_start is not None and current_label is not None and current_end is not None:
                    spans.append(Span(current_start, current_end, current_label, min(current_scores)))
                current_start = start
                current_label = entity
                current_scores = [float(score)]
            else:
                current_scores.append(float(score))

            current_end = end

        if current_start is not None and current_label is not None and current_end is not None:
            spans.append(Span(current_start, current_end, current_label, min(current_scores)))

        return spans


def build_detector(backend: str = "local", threshold: float = 0.5) -> Detector:
    if backend in {"local", "bardsai", "onnx"}:
        return BardsAiOnnxDetector(threshold=threshold)
    raise ValueError(f"Unsupported detector backend: {backend}")


def _dedupe_and_merge(spans: list[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
    merged: list[Span] = []
    for span in ordered:
        if span.start >= span.end:
            continue
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        if previous.label == span.label and span.start <= previous.end + 1:
            merged[-1] = Span(previous.start, max(previous.end, span.end), previous.label, min(previous.score, span.score))
            continue
        if not previous.overlaps(span):
            merged.append(span)
            continue
        if span.end > previous.end:
            merged[-1] = Span(previous.start, span.end, previous.label, max(previous.score, span.score))
    return merged


def _softmax(logits):
    import numpy as np

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
