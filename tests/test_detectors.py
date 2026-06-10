import pytest

from privacy_kit.core.detectors import BardsAiOnnxDetector, build_detector


class FakeEncoding:
    def __init__(self, length: int) -> None:
        self.ids = list(range(length))
        self.attention_mask = [1] * length
        self.type_ids = [0] * length
        self.offsets = [(index, index + 1) for index in range(length)]
        self.word_ids = [None, *range(length - 2), None]


class FakeTokenizer:
    def __init__(self, encoding: FakeEncoding) -> None:
        self.encoding = encoding
        self.truncation_enabled = True

    def no_truncation(self) -> None:
        self.truncation_enabled = False

    def enable_truncation(self, max_length: int, stride: int) -> None:
        self.truncation_enabled = True
        self.max_length = max_length
        self.stride = stride

    def encode(self, text: str) -> FakeEncoding:
        assert text == "long text"
        assert self.truncation_enabled is False
        return self.encoding


def test_bards_detector_chunks_full_tokenization() -> None:
    encoding = FakeEncoding(12)
    detector = object.__new__(BardsAiOnnxDetector)
    detector._tokenizer = FakeTokenizer(encoding)
    detector.max_tokens = 6
    detector.stride = 2

    chunks = detector._chunked_encodings("long text")

    assert [chunk.ids for chunk in chunks] == [
        [0, 1, 2, 3, 4, 11],
        [0, 3, 4, 5, 6, 11],
        [0, 5, 6, 7, 8, 11],
        [0, 7, 8, 9, 10, 11],
    ]


def test_regex_backend_is_not_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported detector backend: regex"):
        build_detector("regex")
