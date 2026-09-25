"""Run and validate only the configured selected-clip tracking experiments.

The scope checks are intentional: this entry point must not silently run the
SportsMOT benchmark or rebuild the main gallery.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "selected_clips_cpu.yaml"
EXPECTED_VIDEOS = {
    "oppie": "data/oppie.mp4",
    "resist": "data/resist.mp4",
    "Joker": "data/Joker.mp4",
    "elonkanye": "data/elonKanye.mp4",
    "saul": "data/saul.mp4",
    "MI6": "data/MI6.mp4",
    "football-1": "data/videos/desktop_clips/football-1.mp4",
    "football-2": "data/videos/desktop_clips/football-2.mp4",
    "football-3": "data/videos/desktop_clips/football-3.mp4",
}
EXPECTED_METHODS = ("ocsort", "deep_ocsort", "deep_ocsort_scenecut")


def validate_selected_scope(config: dict[str, Any]) -> None:
    """Fail closed if the config ever expands beyond the selected gallery."""
    from scenecut_tracking.config import merged_config

    experiment = config.get("experiment", {})
    videos = experiment.get("videos", {})
    methods = tuple(experiment.get("methods", ()))

    if videos != EXPECTED_VIDEOS:
        raise ValueError(
            f"Selected runner is restricted to: {', '.join(EXPECTED_VIDEOS)}"
        )
    if methods != EXPECTED_METHODS:
        raise ValueError(
            "Selected runner requires exactly OC-SORT, Deep OC-SORT, and "
            "Deep OC-SORT + Scene Cut"
        )
    if experiment.get("output_root") != "outputs/selected_clips":
        raise ValueError("Selected runner may write only to outputs/selected_clips")
    if experiment.get("gallery_filename") != "selected_clips_gallery.html":
        raise ValueError("Selected runner may update only selected_clips_gallery.html")
    if experiment.get("manifest_filename") != "selected_clips_manifest.json":
        raise ValueError("Unexpected selected-clips manifest target")
    if experiment.get("process_all_frames") is not True:
        raise ValueError("Selected videos must retain every source frame")
    overrides = experiment.get("video_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(EXPECTED_VIDEOS):
        raise ValueError("Video overrides must name only selected clips")
    if not (PROJECT_ROOT / "yolo11n.pt").is_file():
        raise FileNotFoundError("Local yolo11n.pt is required; no download is allowed")
    for name in EXPECTED_VIDEOS:
        settings = merged_config(config, overrides.get(name, {}))
        detector = settings.get("detector", {})
        if detector.get("backend") != "ultralytics" or detector.get("model") != "yolo11n.pt":
            raise ValueError(f"{name} must use local YOLO11n")
        if detector.get("local_only") is not True:
            raise ValueError(f"{name} must forbid detector downloads")
        reid = settings.get("reid", {})
        if reid.get("local_only", True) is not True:
            raise ValueError(f"{name} must forbid ReID downloads")
        for section in ("deep_ocsort", "reid"):
            weights = PROJECT_ROOT / "weights" / str(settings[section]["weights"])
            if not weights.is_file():
                raise FileNotFoundError(f"{name} requires local {section} weights: {weights}")
        devices = (
            experiment.get("device"),
            detector.get("device"),
            settings.get("deep_ocsort", {}).get("device"),
            settings.get("reid", {}).get("device"),
        )
        if any(str(device).lower() != "cpu" for device in devices):
            raise ValueError("Selected runner is configured for CPU-only execution")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Run the {len(EXPECTED_VIDEOS)} selected clips with all three configured trackers."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute caches and all 15 outputs even when they are already current",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from scenecut_tracking.config import load_config

    config = load_config(CONFIG_PATH)
    validate_selected_scope(config)

    runner = PROJECT_ROOT / "scripts" / "run_all_experiments.py"
    command = [
        sys.executable,
        "-u",
        str(runner),
        "--config",
        str(CONFIG_PATH),
    ]
    if args.force:
        command.append("--force")

    print(f"Scope: {len(EXPECTED_VIDEOS)} selected clips x 3 methods; CPU only", flush=True)
    print("Gallery: outputs/reports/selected_clips_gallery.html", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    validator = PROJECT_ROOT / "scripts" / "validate_outputs.py"
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--config",
            str(CONFIG_PATH),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
