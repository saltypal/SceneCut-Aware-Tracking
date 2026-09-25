"""Regression checks for the local BoxMOT NumPy compatibility patch."""

import numpy as np

from scenecut_tracking.ocsort_tracker import _patch_boxmot_numpy_compatibility


def _filter_with_observation():
    from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR

    kalman_filter = KalmanFilterXYSR(dim_x=7, dim_z=4)
    kalman_filter.H[:4, :4] = np.eye(4)
    first_box = np.array([[50.0], [70.0], [400.0], [1.0]])
    kalman_filter.x[:4] = first_box
    kalman_filter.update(first_box)
    return kalman_filter


def test_unfreeze_after_history_evicted_previous_observation():
    _patch_boxmot_numpy_compatibility()
    kalman_filter = _filter_with_observation()
    kalman_filter.update(None)
    for _ in range(kalman_filter.max_obs):
        kalman_filter.update(None)

    next_box = np.array([[52.0], [72.0], [400.0], [1.0]])
    kalman_filter.update(next_box)

    assert kalman_filter.attr_saved is None
    assert np.isfinite(kalman_filter.x).all()
    assert np.allclose(kalman_filter.z, next_box)


def test_unfreeze_still_interpolates_when_two_observations_exist():
    _patch_boxmot_numpy_compatibility()
    kalman_filter = _filter_with_observation()
    kalman_filter.update(None)

    next_box = np.array([[52.0], [72.0], [400.0], [1.0]])
    kalman_filter.update(next_box)

    assert kalman_filter.attr_saved is None
    assert np.isfinite(kalman_filter.x).all()
    assert np.allclose(kalman_filter.z, next_box)
