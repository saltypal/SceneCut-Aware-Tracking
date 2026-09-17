"""Immutable frame-indexed storage for sharing YOLO detections across experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .frame_source import iter_source_frames, read_source_metadata
from .types import VideoMetadata


class DetectionCache:
    def __init__(self, rows: np.ndarray, metadata: dict):
        self.rows = np.asarray(rows, dtype=np.float32).reshape(-1, 7)
        self.metadata = metadata
        self._by_frame: dict[int, np.ndarray] = {}
        if len(self.rows):
            for frame_index in np.unique(self.rows[:, 0].astype(int)):
                mask = self.rows[:, 0].astype(int) == frame_index
                self._by_frame[int(frame_index)] = self.rows[mask, 1:7].copy()

    def for_frame(self, frame_index: int) -> np.ndarray:
        return self._by_frame.get(frame_index, np.empty((0, 6), dtype=np.float32)).copy()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            rows=self.rows,
            metadata=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DetectionCache":
        cache_path = Path(path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Detection cache not found: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as payload:
            rows = payload["rows"]
            metadata = json.loads(str(payload["metadata"].item()))
        return cls(rows=rows, metadata=metadata)

    def validate_source(self, source_path: str | Path) -> VideoMetadata:
        current = read_source_metadata(source_path)
        cached = self.metadata["video"]
        mismatches = []
        for field in ["fingerprint", "frame_count", "width", "height"]:
            if current.to_dict()[field] != cached[field]:
                mismatches.append(field)
        if mismatches:
            raise ValueError(
                "Detection cache does not belong to this source; mismatched fields: "
                + ", ".join(mismatches)
            )
        return current

    def validate_video(self, video_path: str | Path) -> VideoMetadata:
        """Backward-compatible alias for existing callers and notebooks."""
        return self.validate_source(video_path)


def build_detection_cache(source_path: str | Path, detector, detector_config: dict) -> DetectionCache:
    metadata = read_source_metadata(source_path)
    rows: list[np.ndarray] = []
    frame_count = 0
    started = time.perf_counter()
    for frame_index, frame in iter_source_frames(source_path):
        detections = detector.detect(frame)
        if len(detections):
            frame_column = np.full((len(detections), 1), frame_index, dtype=np.float32)
            rows.append(np.hstack([frame_column, detections]))
        frame_count = frame_index + 1
    if frame_count != metadata.frame_count:
        metadata = VideoMetadata(
            path=metadata.path,
            fingerprint=metadata.fingerprint,
            frame_count=frame_count,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        )
    all_rows = np.vstack(rows) if rows else np.empty((0, 7), dtype=np.float32)
    elapsed = time.perf_counter() - started
    return DetectionCache(
        rows=all_rows,
        metadata={
            "schema_version": 2,
            "video": metadata.to_dict(),
            "detector": detector_config,
            "runtime": {
                "elapsed_seconds": elapsed,
                "processing_fps": frame_count / max(elapsed, 1e-12),
                "detection_rows": int(len(all_rows)),
            },
        },
    )
