"""Pretrained single-class detector adapters used by tracking experiments.

The project defaults to official COCO-pretrained YOLOX-S.  The Ultralytics
adapter remains available for controlled detector comparisons, but a run must
always cache its detections before either tracker experiment begins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import configure_runtime, resolve_device


class YoloPersonDetector:
    """Dispatch to a configured pretrained detector without changing its outputs."""

    def __init__(self, config: dict, project_root: str | Path):
        configure_runtime(project_root)
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.device = resolve_device(str(config.get("device", "auto")))
        self.backend = str(config.get("backend", "ultralytics")).lower()
        self.target_class = int(config.get("target_class", config.get("person_class", 0)))
        if self.backend == "yolox":
            self._load_yolox()
        elif self.backend == "ultralytics":
            model_reference = Path(str(config["model"]))
            if not model_reference.is_absolute():
                project_model = self.project_root / model_reference
                if project_model.is_file():
                    model_reference = project_model
            if config.get("local_only", False) and not model_reference.is_file():
                raise FileNotFoundError(f"Local detector weights are missing: {model_reference}")

            from ultralytics import YOLO

            self.model = YOLO(str(model_reference))
        else:
            raise ValueError(f"Unsupported detector backend: {self.backend}")

    def _load_yolox(self) -> None:
        """Load the official YOLOX implementation and checkpoint from local paths."""
        import sys
        import torch

        yolox_root = self.project_root / ".runtime" / "YOLOX"
        checkpoint_path = self.project_root / "weights" / str(self.config["weights"])
        if not yolox_root.is_dir():
            raise FileNotFoundError(
                "Official YOLOX source is missing. Run the project setup command in README first."
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"YOLOX checkpoint is missing: {checkpoint_path}. Run the project setup command first."
            )
        if str(yolox_root) not in sys.path:
            sys.path.insert(0, str(yolox_root))
        from yolox.exp import get_exp

        experiment = get_exp(None, str(self.config["model"]))
        self.model = experiment.get_model().to(self.device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.yolox_experiment = experiment

    def _detect_yolox(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        image_size = int(self.config["image_size"])
        transformed, resize_ratio = preproc(frame, (image_size, image_size))
        tensor = torch.from_numpy(transformed).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            raw_output = self.model(tensor)
            output = postprocess(
                raw_output,
                num_classes=int(self.yolox_experiment.num_classes),
                conf_thre=float(self.config["confidence"]),
                nms_thre=float(self.config["iou"]),
                class_agnostic=True,
            )[0]
        if output is None or len(output) == 0:
            return np.empty((0, 6), dtype=np.float32)
        rows = output.detach().cpu().numpy()
        boxes = rows[:, :4] / max(float(resize_ratio), 1e-12)
        confidence = (rows[:, 4] * rows[:, 5]).reshape(-1, 1)
        classes = rows[:, 6].reshape(-1, 1)
        selected = classes[:, 0] == self.target_class
        detections = np.hstack([boxes[selected], confidence[selected], classes[selected]])
        max_detections = int(self.config.get("max_detections", len(detections)))
        return detections[np.argsort(-detections[:, 4])[:max_detections]].astype(np.float32)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if self.backend == "yolox":
            return self._detect_yolox(frame)
        result = self.model.predict(
            source=frame,
            conf=float(self.config["confidence"]),
            iou=float(self.config["iou"]),
            imgsz=int(self.config["image_size"]),
            classes=[self.target_class],
            max_det=int(self.config.get("max_detections", 300)),
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        classes = result.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
        return np.hstack([boxes, confidence, classes]).astype(np.float32)
