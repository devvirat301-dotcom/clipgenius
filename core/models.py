"""Plain data classes shared by all modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Moment:
    start: float
    end: float
    title: str
    category: str
    reason: str
    scores: dict[str, float]
    overall: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def segments_to_json(segments: list[Segment]) -> list[dict]:
    return [asdict(s) for s in segments]


def segments_from_json(data: list[dict]) -> list[Segment]:
    return [
        Segment(
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(**w) for w in d.get("words", [])],
        )
        for d in data
    ]
