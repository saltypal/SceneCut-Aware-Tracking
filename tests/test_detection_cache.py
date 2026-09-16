from pathlib import Path

import cv2
import numpy as np

from scenecut_tracking.detection_cache import DetectionCache, build_detection_cache


class FakeDetector:
    def detect(self, frame):
        return np.asarray([[5, 6, 20, 30, 0.9, 0]], dtype=np.float32)


def _video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    assert writer.isOpened()
    for value in [10, 30, 50]:
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def test_detection_cache_round_trip_and_video_validation(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    cache_path = tmp_path / "tiny.npz"
    _video(video_path)
    cache = build_detection_cache(video_path, FakeDetector(), {"model": "fake"})
    cache.save(cache_path)

    loaded = DetectionCache.load(cache_path)
    metadata = loaded.validate_video(video_path)

    assert metadata.frame_count == 3
    assert loaded.rows.shape == (3, 7)
    np.testing.assert_array_equal(cache.rows, loaded.rows)

