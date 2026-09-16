"""OSNet person appearance embedding adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import configure_runtime, resolve_device


class OSNetEmbedder:
    def __init__(self, config: dict, project_root: str | Path):
        configure_runtime(project_root)
        from boxmot.reid.core.reid_handler import ReID

        self.config = config
        self.device = resolve_device(str(config.get("device", "auto")))
        weights = Path(project_root) / "weights" / str(config["weights"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = ReID(weights=weights, device=self.device, half=False)

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.empty((0, 0), dtype=np.float32)
        features = np.asarray(
            self.model(frame, np.hstack([boxes, np.ones((len(boxes), 2), dtype=np.float32)])),
            dtype=np.float32,
        )
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)

