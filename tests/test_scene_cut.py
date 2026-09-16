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

