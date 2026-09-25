"""Missing weights must fail before a model library can fetch them."""

import pytest

from scenecut_tracking.reid import OSNetEmbedder


def test_local_reid_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local ReID weights are missing"):
        OSNetEmbedder(
            {"weights": "missing_osnet.pt", "device": "cpu"},
            tmp_path,
        )
