from pathlib import Path

from scenecut_tracking.config import load_config
from scenecut_tracking.video_gallery import build_video_gallery


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selected_config_has_nine_requested_sources_and_three_methods():
    config = load_config(PROJECT_ROOT / "configs" / "selected_clips_cpu.yaml")
    assert list(config["experiment"]["videos"]) == [
        "oppie", "resist", "Joker", "elonkanye", "saul",
        "MI6", "football-1", "football-2", "football-3",
    ]
    assert list(config["experiment"]["methods"]) == [
        "ocsort", "deep_ocsort", "deep_ocsort_scenecut"
    ]
    assert config["detector"]["device"] == "cpu"
    assert config["deep_ocsort"]["device"] == "cpu"
    for source in config["experiment"]["videos"].values():
        assert (PROJECT_ROOT / source).is_file()


def test_selected_gallery_contains_only_requested_clips_and_three_methods(tmp_path):
    result_dir = tmp_path / "outputs" / "selected_clips" / "oppie" / "ocsort"
    result_dir.mkdir(parents=True)
    (result_dir / "annotated_h264.mp4").touch()

    gallery = build_video_gallery(
        tmp_path,
        video_names=("oppie", "Joker"),
        method_keys=("ocsort", "deep_ocsort", "deep_ocsort_scenecut"),
        output_root="outputs/selected_clips",
        output_name="selected_clips_gallery.html",
    )
    page = gallery.read_text(encoding="utf-8")
    assert "2 source videos × 3 tracking methods" in page
    assert "1 of 6 videos complete" in page
    assert "../selected_clips/oppie/ocsort/annotated_h264.mp4" in page
    assert "Scene Cut + OSNet ReID" in page
    assert "OSNet weight 1.0" not in page
    assert "football-1" not in page
    assert "Final clip" not in page
