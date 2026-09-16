import numpy as np

from scenecut_tracking.identity_memory import IdentityMemory


def test_cut_recovery_preserves_global_identity_and_enforces_threshold():
    memory = IdentityMemory(similarity_threshold=0.8, recovery_window_frames=3, gallery_size=4)
    first_ids, _ = memory.assign(0, [1, 2], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert first_ids == [1, 2]

    memory.begin_cut(frame_index=10)
    recovered_ids, decisions = memory.assign(
        10,
        [1, 2, 3],
        np.asarray([[0.99, 0.01], [0.01, 0.99], [-1.0, 0.0]]),
    )

    assert recovered_ids[:2] == [1, 2]
    assert recovered_ids[2] == 3
    assert sum(decision.decision == "recovered" for decision in decisions) == 2


def test_one_old_identity_cannot_be_claimed_twice():
    memory = IdentityMemory(similarity_threshold=0.7, recovery_window_frames=3, gallery_size=4)
    memory.assign(0, [1], np.asarray([[1.0, 0.0]]))
    memory.begin_cut(frame_index=5)

    identities, _ = memory.assign(5, [1, 2], np.asarray([[1.0, 0.0], [0.99, 0.01]]))

    assert identities.count(1) == 1
    assert len(set(identities)) == 2

