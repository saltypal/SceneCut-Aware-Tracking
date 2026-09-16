import numpy as np

from scenecut_tracking.ocsort_tracker import OCSortTracker


TRACKER_CONFIG = {
    "detection_threshold": 0.25,
    "max_age": 30,
    "min_hits": 1,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "inertia": 0.2,
    "use_byte": False,
}


def test_tracker_reset_discards_motion_state():
    tracker = OCSortTracker(TRACKER_CONFIG)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = np.asarray([[20, 20, 60, 90, 0.95, 0]], dtype=np.float32)
    tracks = tracker.update(detections, frame)
    assert len(tracks) == 1
    assert tracker.active_track_count == 1

    tracker.reset_motion_state()

    assert tracker.generation == 1
    assert tracker.active_track_count == 0

