"""Shared value objects used across detection, tracking, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fingerprint: str
    frame_count: int
    width: int
    height: int
    fps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackRecord:
    frame: int
    local_id: int
    global_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    detection_index: int
    tracker_generation: int

    def to_mot_row(self) -> list[float]:
        return [
            self.frame + 1,
            self.global_id,
            self.x1,
            self.y1,
            self.x2 - self.x1,
            self.y2 - self.y1,
            self.confidence,
            self.class_id,
            1.0,
        ]


@dataclass(frozen=True)
class RecoveryDecision:
    frame: int
    local_id: int
    global_id: int
    decision: str
    similarity: float | None

