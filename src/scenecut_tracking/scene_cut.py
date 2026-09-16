"""Streaming hard-cut detectors."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CutDecision:
    is_cut: bool
    score: float | None
    backend: str


class PySceneDetectCutDetector:
    def __init__(self, threshold: float, min_scene_length: int, fps: float):
        from scenedetect import FrameTimecode
        from scenedetect.detector import FlashFilter
        from scenedetect.detectors import ContentDetector

        self.fps = float(fps)
        self.frame_timecode_type = FrameTimecode
        self.detector = ContentDetector(
            threshold=float(threshold),
            min_scene_len=int(min_scene_length),
            filter_mode=FlashFilter.Mode.SUPPRESS,
        )

    def update(self, frame_index: int, frame: np.ndarray) -> CutDecision:
        timecode = self.frame_timecode_type(frame_index, fps=self.fps)
        cuts = self.detector.process_frame(timecode, frame)
        cut_frames = {int(cut.frame_num) for cut in cuts}
        score = getattr(self.detector, "_frame_score", None)
        return CutDecision(frame_index in cut_frames, None if score is None else float(score), "pyscenedetect")


class HistogramCutDetector:
    def __init__(self, threshold: float, min_scene_length: int):
        self.threshold = float(threshold)
        self.min_scene_length = int(min_scene_length)
        self.previous_histogram: np.ndarray | None = None
        self.last_cut_frame = -self.min_scene_length

    @staticmethod
    def _histogram(frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256]
        )
        return cv2.normalize(histogram, histogram).flatten()

    def update(self, frame_index: int, frame: np.ndarray) -> CutDecision:
        current = self._histogram(frame)
        if self.previous_histogram is None:
            self.previous_histogram = current
            return CutDecision(False, 0.0, "histogram")
        score = float(cv2.compareHist(self.previous_histogram, current, cv2.HISTCMP_BHATTACHARYYA))
        self.previous_histogram = current
        enough_gap = frame_index - self.last_cut_frame >= self.min_scene_length
        is_cut = enough_gap and score >= self.threshold
        if is_cut:
            self.last_cut_frame = frame_index
        return CutDecision(is_cut, score, "histogram")


def create_scene_cut_detector(config: dict, fps: float):
    backend = str(config.get("backend", "pyscenedetect")).lower()
    if backend == "histogram":
        return HistogramCutDetector(
            threshold=config["histogram_threshold"],
            min_scene_length=config["min_scene_length"],
        )
    if backend == "pyscenedetect":
        return PySceneDetectCutDetector(
            threshold=config["content_threshold"],
            min_scene_length=config["min_scene_length"],
            fps=fps,
        )
    raise ValueError(f"Unknown scene-cut backend: {backend}")
