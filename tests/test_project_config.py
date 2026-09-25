from pathlib import Path

from scenecut_tracking.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_both_project_configs_parse_and_use_local_yolo_weights():
    for name in ("default.yaml", "ultimate_cpu.yaml"):
        config = load_config(PROJECT_ROOT / "configs" / name)
        assert config["detector"]["model"] == "yolo11n.pt"
        assert config["detector"]["device"] == "cpu"


def test_canonical_config_has_a_real_byte_confidence_band():
    config = load_config(PROJECT_ROOT / "configs" / "ultimate_cpu.yaml")
    assert config["tracker"]["use_byte"] is True
    assert config["detector"]["confidence"] < config["tracker"]["detection_threshold"]
    assert config["scene_cut"]["hard_cut_histogram_threshold"] > 0
