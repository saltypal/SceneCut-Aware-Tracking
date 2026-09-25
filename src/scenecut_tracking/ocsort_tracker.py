"""Adapter around BoxMOT's OC-SORT implementation."""

from __future__ import annotations

import numpy as np

from .runtime import configure_runtime


def _patch_boxmot_numpy_compatibility() -> None:
    """Keep BoxMOT's OC-SORT observation re-update working with NumPy 2.x.

    BoxMOT stores four-dimensional measurements as ``(4, 1)`` arrays. Its
    upstream online-smoothing method converts a one-element array via
    ``float(array)``; NumPy 2 rejects that conversion. The replacement is the
    same algorithm with one explicit scalar extraction at that boundary.
    """
    from collections import deque
    from copy import deepcopy

    from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR

    if getattr(KalmanFilterXYSR, "_scenecut_numpy_compat", False):
        return

    def compatible_unfreeze(self) -> None:
        if self.attr_saved is None:
            return
        new_history = deepcopy(list(self.history_obs))
        observed_indices = np.flatnonzero([item is not None for item in new_history])
        if len(observed_indices) < 2:
            # A long gap can evict the previous observation from the bounded
            # history. There is no pair of boxes to interpolate, so retain the
            # Kalman prediction and let the incoming measurement update it.
            self.attr_saved = None
            return
        self.__dict__ = self.attr_saved
        self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
        index1, index2 = observed_indices[-2], observed_indices[-1]
        box1, box2 = new_history[index1], new_history[index2]
        x1, y1, s1, r1 = box1
        x2, y2, s2, r2 = box2
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
        for step in range(index2 - index1):
            x, y = x1 + (step + 1) * dx, y1 + (step + 1) * dy
            w, h = w1 + (step + 1) * dw, h1 + (step + 1) * dh
            area = w * h
            ratio = w / np.asarray(h).reshape(-1)[0].item()
            self.update(np.asarray([x, y, area, ratio]).reshape((4, 1)))
            if step != index2 - index1 - 1:
                self.predict()
                self.history_obs.pop()
        self.history_obs.pop()

    KalmanFilterXYSR.unfreeze = compatible_unfreeze
    KalmanFilterXYSR._scenecut_numpy_compat = True


class OCSortTracker:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.generation = 0
        self._tracker = self._create_tracker()

    def _create_tracker(self):
        configure_runtime(__file__.rsplit("src", 1)[0])
        _patch_boxmot_numpy_compatibility()
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
