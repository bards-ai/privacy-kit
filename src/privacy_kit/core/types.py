from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float = 1.0

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end
