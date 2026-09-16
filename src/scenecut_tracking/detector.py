"""Pretrained YOLO person detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import configure_runtime, resolve_device


class YoloPersonDetector:
    """Thin, deterministic adapter around Ultralytics prediction mode."""

    def __init__(self, config: dict, project_root: str | Path):
        configure_runtime(project_root)
        from ultralytics import YOLO

        self.config = config
        self.device = resolve_device(str(config.get("device", "auto")))
        self.model = YOLO(str(config["model"]))

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            source=frame,
            conf=float(self.config["confidence"]),
            iou=float(self.config["iou"]),
            imgsz=int(self.config["image_size"]),
            classes=[int(self.config["person_class"])],
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        classes = result.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
        return np.hstack([boxes, confidence, classes]).astype(np.float32)

