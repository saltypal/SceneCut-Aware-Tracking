"""Adapter around BoxMOT's OC-SORT implementation."""

from __future__ import annotations

import numpy as np

from .runtime import configure_runtime


class OCSortTracker:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.generation = 0
        self._tracker = self._create_tracker()

    def _create_tracker(self):
        configure_runtime(__file__.rsplit("src", 1)[0])
        from boxmot import OcSort

        return OcSort(
            det_thresh=float(self.config["detection_threshold"]),
            min_conf=min(0.10, float(self.config["detection_threshold"])),
            max_age=int(self.config["max_age"]),
            min_hits=int(self.config["min_hits"]),
            iou_threshold=float(self.config["iou_threshold"]),
            delta_t=int(self.config["delta_t"]),
            inertia=float(self.config["inertia"]),
            use_byte=bool(self.config["use_byte"]),
            per_class=False,
            asso_func="iou",
        )

    def reset_motion_state(self) -> None:
        """Discard all Kalman and tracklet state while keeping generation explicit."""
        self.generation += 1
        self._tracker = self._create_tracker()

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        tracks = self._tracker.update(detections, frame)
        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(tracks, dtype=np.float32).reshape(-1, 8)

    @property
    def active_track_count(self) -> int:
        return len(self._tracker.active_tracks)
