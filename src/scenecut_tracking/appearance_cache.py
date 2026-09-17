"""Immutable per-detection OSNet embeddings shared by Deep OC-SORT methods."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .detection_cache import DetectionCache
from .frame_source import iter_source_frames
from .reid import OSNetEmbedder


class AppearanceCache:
    def __init__(self, embeddings: np.ndarray, offsets: np.ndarray, metadata: dict):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.metadata = metadata
        if self.offsets.ndim != 1 or len(self.offsets) < 1:
            raise ValueError("Appearance offsets must be a non-empty one-dimensional array")
        if int(self.offsets[-1]) != len(self.embeddings):
            raise ValueError("Appearance offsets do not cover every embedding row")

    def for_frame(self, frame_index: int) -> np.ndarray:
        start = int(self.offsets[frame_index])
        end = int(self.offsets[frame_index + 1])
        return self.embeddings[start:end].copy()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            embeddings=self.embeddings,
            offsets=self.offsets,
            metadata=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AppearanceCache":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            return cls(
                embeddings=payload["embeddings"],
                offsets=payload["offsets"],
                metadata=json.loads(str(payload["metadata"].item())),
            )

    def validate(self, detection_cache: DetectionCache) -> None:
        expected = detection_cache.metadata["video"]
        actual = self.metadata["video"]
        for field in ("fingerprint", "frame_count", "width", "height"):
            if actual[field] != expected[field]:
                raise ValueError(f"Appearance cache video mismatch: {field}")
        if self.metadata["detection_rows"] != len(detection_cache.rows):
            raise ValueError("Appearance cache detection-row count does not match")
        if len(self.offsets) != expected["frame_count"] + 1:
            raise ValueError("Appearance cache does not cover every frame")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_appearance_cache(
    video_path: str | Path,
    detection_cache_path: str | Path,
    reid_config: dict,
    project_root: str | Path,
) -> AppearanceCache:
    """Extract OSNet once for every cached detection, including all empty frames."""
    root = Path(project_root).resolve()
    detections = DetectionCache.load(detection_cache_path)
    metadata = detections.validate_source(video_path)
    embedder = OSNetEmbedder(reid_config, root)
    chunks: list[np.ndarray] = []
    offsets = [0]
    embedding_width = 0
    processed_frames = 0
    started = time.perf_counter()

    for frame_index, frame in iter_source_frames(video_path):
        frame_detections = detections.for_frame(frame_index)
        if len(frame_detections):
            frame_embeddings = embedder.extract(frame, frame_detections[:, :4])
            if len(frame_embeddings) != len(frame_detections):
                raise RuntimeError("OSNet did not return one embedding per detection")
            embedding_width = frame_embeddings.shape[1]
            chunks.append(frame_embeddings.astype(np.float32))
            offsets.append(offsets[-1] + len(frame_embeddings))
        else:
            offsets.append(offsets[-1])
        processed_frames = frame_index + 1

    if processed_frames != metadata.frame_count:
        raise RuntimeError(
            f"Appearance cache processed {processed_frames}/{metadata.frame_count} frames"
        )
    all_embeddings = (
        np.vstack(chunks)
        if chunks
        else np.empty((0, embedding_width), dtype=np.float32)
    )
    if len(all_embeddings) != len(detections.rows):
        raise RuntimeError("Appearance cache row count does not match detection cache")

    elapsed = time.perf_counter() - started
    weights = root / "weights" / str(reid_config["weights"])
    return AppearanceCache(
        embeddings=all_embeddings,
        offsets=np.asarray(offsets, dtype=np.int64),
        metadata={
            "schema_version": 1,
            "video": metadata.to_dict(),
            "detection_rows": len(detections.rows),
            "embedding_width": embedding_width,
            "reid": {
                "weights": str(weights),
                "weights_sha256": _sha256(weights),
                "device": str(reid_config.get("device", "cpu")),
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "processing_fps": processed_frames / max(elapsed, 1e-12),
            },
        },
    )

