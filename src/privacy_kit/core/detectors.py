"""On-device PII detection.

Loads ``bardsai/eu-pii-anonimization-multilang`` with the dependency-light
stack (``huggingface-hub`` + ``tokenizers`` + ``onnxruntime``) and turns raw
text into character-offset :class:`~privacy_kit.core.types.Span` objects.

Long inputs are processed in overlapping token windows (batched through the
model) so PII past the 512-token limit is never silently dropped. Subword
predictions are aggregated to whole words to prevent partial-token leakage
(e.g. "ample.com" surviving because only "ex" was tagged).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

from privacy_kit.core.types import Span

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MODEL_ID = "bardsai/eu-pii-anonimization-multilang"

# How many overlapping windows go through the model in one forward pass. Keeps
# long-input latency down without letting the padded batch tensor grow unbounded.
_BATCH_WINDOWS = 8


class Detector(Protocol):
    """Anything that can find PII spans in text."""

    def detect(self, text: str) -> list[Span]: ...


def trim_spans(text: str, spans: list[Span]) -> list[Span]:
    """Trim trailing non-alphanumeric characters from each span.

    Word-level aggregation captures attached punctuation (e.g. the comma in
    "Anna," when only "Anna" is the entity). Masking the punctuation is
    harmless but ugly and slightly hurts placeholder stability, so trim it.
    Spans that become empty are dropped.
    """
    trimmed: list[Span] = []
    for span in spans:
        end = span.end
        while end > span.start and not text[end - 1].isalnum():
            end -= 1
        if end > span.start:
            trimmed.append(Span(span.start, end, span.label, span.score))
    return trimmed


class BardsAiOnnxDetector:
    """Local ONNX Runtime backend for bardsai/eu-pii-anonimization-multilang.

    Construct once and reuse — model loading is expensive. ``max_tokens`` is
    the model's sequence limit including the two special tokens; ``stride`` is
    the overlap between consecutive windows on long inputs.
    """

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float = 0.5,
        max_tokens: int = 512,
        stride: int = 128,
        intra_op_num_threads: int | None = None,
        graph_optimization: bool = True,
    ) -> None:
        if stride >= max_tokens - 2:
            raise ValueError("stride must be smaller than max_tokens - 2")
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
                "PII redaction model dependencies are missing. "
                "Install the package with `pip install privacy-kit`."
            ) from exc

        resolved_model_id = model_id or os.getenv("PII_MODEL_ID", DEFAULT_MODEL_ID)
        cache_dir = os.getenv("PII_MODEL_CACHE_DIR")

        try:
            tokenizer_path = hf_hub_download(
                resolved_model_id, "tokenizer.json", cache_dir=cache_dir
            )
            config_path = hf_hub_download(resolved_model_id, "config.json", cache_dir=cache_dir)
            model_path = hf_hub_download(
                resolved_model_id, "onnx/model_quantized.onnx", cache_dir=cache_dir
            )
        except Exception as exc:
            cache_hint = f" using cache dir {cache_dir!r}" if cache_dir else ""
            raise RuntimeError(
                f"Could not download or load model files for {resolved_model_id!r}{cache_hint}. "
                "Check network access, Hugging Face availability, "
                "or set PII_MODEL_CACHE_DIR to a writable directory."
            ) from exc

        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)

        self.id2label = {int(key): value for key, value in config["id2label"].items()}
        self._np = np
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # The shipped tokenizer.json carries a 512-token truncation config;
        # windowing handles long inputs, so encoding must never truncate.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

        specials = self._tokenizer.encode("").ids
        if len(specials) != 2:
            raise RuntimeError(
                f"Expected exactly <cls> and <sep> when encoding empty text, got {specials!r}."
            )
        self._cls_id, self._sep_id = specials
        pad_id = config.get("pad_token_id")
        if pad_id is None:
            pad_id = self._tokenizer.token_to_id("<pad>")
        self._pad_id = int(pad_id) if pad_id is not None else 0

        session_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            session_options.intra_op_num_threads = intra_op_num_threads
        if not graph_optimization:
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
                "Check that onnxruntime supports this host "
                "and that the cached model file is not corrupted."
            ) from exc
        self._input_names = {model_input.name for model_input in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

    @property
    def labels(self) -> list[str]:
        """The canonical entity-type set (BIO prefixes stripped), sorted."""
        return sorted({lbl[2:] for lbl in self.id2label.values() if lbl != "O"})

    def warmup(self) -> None:
        """Run one tiny inference so the first real request pays no init latency.

        The first forward pass initializes lazy backend state (ONNX session
        memory arenas); doing it at startup keeps it off the hot path.
        """
        self.detect("privacy-kit warmup ping.")

    def detect(self, text: str) -> list[Span]:
        """Return PII spans in ``text`` above the configured score threshold."""
        if not text.strip():
            return []

        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = encoding.ids
        offsets: list[tuple[int, int]] = encoding.offsets
        word_ids: list[int | None] = encoding.word_ids
        if not ids:
            return []

        label_ids, scores = self._best_predictions(ids)
        spans = self._aggregate(offsets, word_ids, label_ids, scores)
        return self._postprocess(text, spans)

    def _plan_windows(self, token_count: int) -> list[int]:
        """Start indices of the overlapping windows covering ``token_count`` tokens."""
        window = self.max_tokens - 2
        step = max(window - self.stride, 1)
        starts: list[int] = []
        for start in range(0, token_count, step):
            starts.append(start)
            if start + window >= token_count:
                break
        return starts

    def _best_predictions(self, ids: list[int]) -> tuple[list[int | None], list[float]]:
        """Run overlapping windows through the model, batched.

        For tokens covered by several windows, keep the most confident
        prediction per global token index.
        """
        np = self._np
        n = len(ids)
        window = self.max_tokens - 2
        best_lid: list[int | None] = [None] * n
        best_score: list[float] = [0.0] * n

        window_starts = self._plan_windows(n)
        for bstart in range(0, len(window_starts), _BATCH_WINDOWS):
            batch = window_starts[bstart : bstart + _BATCH_WINDOWS]
            chunks = [ids[s : s + window] for s in batch]
            width = max(len(c) for c in chunks) + 2
            input_ids = np.array(
                [
                    [self._cls_id, *c, self._sep_id] + [self._pad_id] * (width - len(c) - 2)
                    for c in chunks
                ],
                dtype=np.int64,
            )
            attention_mask = np.array(
                [[1] * (len(c) + 2) + [0] * (width - len(c) - 2) for c in chunks],
                dtype=np.int64,
            )
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self._input_names:
                inputs["token_type_ids"] = np.zeros_like(input_ids)

            logits = self._session.run([self._output_name], inputs)[0]
            probabilities = _softmax(logits)
            for row, (start, chunk) in enumerate(zip(batch, chunks, strict=True)):
                row_probs = probabilities[row][1 : len(chunk) + 1]  # drop <s>, </s>, padding
                lids = row_probs.argmax(axis=-1)
                scores = row_probs.max(axis=-1)
                for j in range(len(chunk)):
                    gi = start + j
                    score = float(scores[j])
                    if best_lid[gi] is None or score > best_score[gi]:
                        best_lid[gi] = int(lids[j])
                        best_score[gi] = score

        return best_lid, best_score

    def _aggregate(
        self,
        offsets: Sequence[tuple[int, int]],
        word_ids: Sequence[int | None],
        label_ids: Sequence[int | None],
        scores: Sequence[float],
    ) -> list[Span]:
        """Aggregate subword predictions to whole words, then merge same-type runs.

        Tagging the *whole word* whenever any of its subwords is PII prevents
        partial-token leakage. Consecutive words of the same entity type are
        then merged into a single span covering the gap (usually a space)
        between them.
        """
        # Group token indices into words (skip zero-width and special tokens).
        word_tokens: list[list[int]] = []
        cur_wid: int | None = None
        for idx, wid in enumerate(word_ids):
            if wid is None or offsets[idx][0] == offsets[idx][1]:
                continue
            if not word_tokens or wid != cur_wid:
                word_tokens.append([idx])
                cur_wid = wid
            else:
                word_tokens[-1].append(idx)

        # Decide each word's entity type from its highest-scoring non-O subword.
        words: list[tuple[int, int, str | None, float]] = []
        for toks in word_tokens:
            cstart = min(offsets[t][0] for t in toks)
            cend = max(offsets[t][1] for t in toks)
            best: tuple[str, float] | None = None
            for t in toks:
                lid = label_ids[t]
                label = self.id2label.get(lid, "O") if lid is not None else "O"
                if label == "O":
                    continue
                etype = label.partition("-")[2]
                if best is None or scores[t] > best[1]:
                    best = (etype, scores[t])
            words.append((cstart, cend, best[0] if best else None, best[1] if best else 0.0))

        # Merge consecutive words sharing an entity type into one span.
        spans: list[Span] = []
        cur_type: str | None = None
        cur_start = 0
        cur_end = 0
        cur_scores: list[float] = []

        def flush() -> None:
            if cur_type is not None and cur_scores:
                spans.append(Span(cur_start, cur_end, cur_type, sum(cur_scores) / len(cur_scores)))

        for w_start, w_end, w_type, w_score in words:
            if w_type is None:
                flush()
                cur_type = None
            elif w_type == cur_type:
                cur_end = w_end
                cur_scores.append(w_score)
            else:
                flush()
                cur_type = w_type
                cur_start = w_start
                cur_end = w_end
                cur_scores = [w_score]
        flush()
        return spans

    def _postprocess(self, text: str, spans: list[Span]) -> list[Span]:
        """Drop low-confidence spans, trim punctuation, de-duplicate overlaps."""
        kept = [s for s in spans if s.score >= self.threshold]
        kept = trim_spans(text, kept)
        # Longest-first within a start position so overlaps keep the widest span.
        kept.sort(key=lambda s: (s.start, -(s.end - s.start)))
        result: list[Span] = []
        for span in kept:
            if result and span.start < result[-1].end:
                continue  # overlapped by an already-kept span
            result.append(span)
        return result


def build_detector(backend: str = "local", threshold: float = 0.5) -> Detector:
    if backend in {"local", "bardsai", "onnx"}:
        return BardsAiOnnxDetector(threshold=threshold)
    raise ValueError(f"Unsupported detector backend: {backend}")


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
