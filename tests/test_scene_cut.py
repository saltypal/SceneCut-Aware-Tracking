import numpy as np

from scenecut_tracking.scene_cut import HistogramCutDetector, PySceneDetectCutDetector


def test_histogram_detector_flags_abrupt_color_change():
    detector = HistogramCutDetector(threshold=0.4, min_scene_length=2)
    dark = np.zeros((80, 120, 3), dtype=np.uint8)
    bright = np.full((80, 120, 3), 255, dtype=np.uint8)

    assert detector.update(0, dark).is_cut is False
    assert detector.update(1, dark).is_cut is False
    assert detector.update(2, bright).is_cut is True


def test_pyscenedetect_detector_flags_abrupt_change():
    detector = PySceneDetectCutDetector(threshold=10.0, min_scene_length=2, fps=10.0)
    dark = np.zeros((80, 120, 3), dtype=np.uint8)
    bright = np.full((80, 120, 3), 255, dtype=np.uint8)

    assert detector.update(0, dark).is_cut is False
    assert detector.update(1, dark).is_cut is False
    assert detector.update(2, bright).is_cut is True


def test_hard_cut_guard_rejects_motion_spike_without_suppressing_next_real_cut():
    detector = PySceneDetectCutDetector(
        threshold=10.0,
        min_scene_length=2,
        fps=10.0,
        hard_cut_histogram_threshold=0.40,
        hard_cut_pixel_threshold=0.12,
    )
    first = np.zeros((80, 120, 3), dtype=np.uint8)
    first[:, :60] = 255
    moved = np.zeros_like(first)
    moved[:, 60:] = 255  # Large pixel motion, but exactly the same color histogram.
    new_scene = np.full_like(first, (0, 0, 255))

    assert detector.update(0, first).is_cut is False
    motion_decision = detector.update(1, moved)
    assert motion_decision.is_cut is False
    assert motion_decision.histogram_distance < 0.40
    assert detector.update(2, new_scene).is_cut is True
