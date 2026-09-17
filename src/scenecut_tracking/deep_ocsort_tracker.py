"""Deep OC-SORT adapter with an explicit hard-cut motion reset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ocsort_tracker import _patch_boxmot_numpy_compatibility
from .runtime import configure_runtime, resolve_device


class DeepOCSortTracker:
    """Wrap BoxMOT Deep OC-SORT and expose embeddings for cross-cut memory."""

    def __init__(self, tracker_config: dict, deep_config: dict, project_root: str | Path):
        self.tracker_config = dict(tracker_config)
        self.deep_config = dict(deep_config)
        self.project_root = Path(project_root).resolve()
        self.generation = 0
        self._tracker = self._create_tracker()

    def _create_tracker(self):
        configure_runtime(self.project_root)
        _patch_boxmot_numpy_compatibility()
        from boxmot import DeepOcSort

        weights = self.project_root / "weights" / str(self.deep_config["weights"])
        if not weights.is_file():
            raise FileNotFoundError(f"Deep OC-SORT ReID checkpoint not found: {weights}")
        device = resolve_device(str(self.deep_config.get("device", "cpu")))
        return DeepOcSort(
            reid_weights=weights,
            device=device,
            half=bool(self.deep_config.get("half", False)),
            det_thresh=float(self.tracker_config["detection_threshold"]),
            max_age=int(self.tracker_config["max_age"]),
            min_hits=int(self.tracker_config["min_hits"]),
            iou_threshold=float(self.tracker_config["iou_threshold"]),
            delta_t=int(self.tracker_config["delta_t"]),
            inertia=float(self.tracker_config["inertia"]),
            w_association_emb=float(self.deep_config["w_association_emb"]),
            alpha_fixed_emb=float(self.deep_config["alpha_fixed_emb"]),
            aw_param=float(self.deep_config["aw_param"]),
            embedding_off=bool(self.deep_config["embedding_off"]),
            cmc_off=bool(self.deep_config["cmc_off"]),
            aw_off=bool(self.deep_config["aw_off"]),
            per_class=False,
            asso_func="iou",
        )

    def reset_motion_state(self) -> None:
        """Discard short-term tracking/CMC state without reloading the ReID model."""
        from boxmot.trackers.deepocsort.deepocsort import KalmanBoxTracker

        self.generation += 1
        self._tracker.active_tracks.clear()
        if self._tracker.per_class_active_tracks is not None:
            self._tracker.per_class_active_tracks.clear()
        self._tracker.frame_count = 0
        self._tracker.cmc = type(self._tracker.cmc)()
        KalmanBoxTracker.count = 1

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
        detection_embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update tracks and return the matching OSNet embedding for every output row."""
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        if detection_embeddings is not None:
            detection_embeddings = np.asarray(detection_embeddings, dtype=np.float32)
            if len(detection_embeddings) != len(detections):
                raise ValueError("Each detection must have one supplied appearance embedding")
        elif len(detections):
            detection_embeddings = np.asarray(
                self._tracker.model.get_features(detections[:, :4], frame), dtype=np.float32
            )
        else:
            detection_embeddings = np.empty((0, 0), dtype=np.float32)

        tracks = self._tracker.update(detections, frame, embs=detection_embeddings)
        if tracks is None or len(tracks) == 0:
            embedding_width = detection_embeddings.shape[1] if detection_embeddings.ndim == 2 else 0
            return (
                np.empty((0, 8), dtype=np.float32),
                np.empty((0, embedding_width), dtype=np.float32),
            )

        track_rows = np.asarray(tracks, dtype=np.float32).reshape(-1, 8)
        detection_indices = track_rows[:, 7].astype(int)
        if np.any(detection_indices < 0) or np.any(detection_indices >= len(detection_embeddings)):
            raise RuntimeError("Deep OC-SORT returned an invalid detection index")
        track_embeddings = detection_embeddings[detection_indices]
        norms = np.linalg.norm(track_embeddings, axis=1, keepdims=True)
        track_embeddings = track_embeddings / np.maximum(norms, 1e-12)
        return track_rows, track_embeddings.astype(np.float32)

    @property
    def active_track_count(self) -> int:
        return len(self._tracker.active_tracks)
