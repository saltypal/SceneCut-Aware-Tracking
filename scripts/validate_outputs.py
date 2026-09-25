"""Check a configured experiment's full-frame artifacts after a rerun.

This validates video containers and decodes every exported frame. It does not
measure tracking accuracy; identity ground truth is needed for that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenecut_tracking.config import load_config


def video_properties(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"Cannot open video: {path}")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frames = 0
    while capture.grab():
        decoded_frames += 1
    capture.release()
    if expected_frames != decoded_frames:
        raise AssertionError(
            f"Container reports {expected_frames} frames but decoded {decoded_frames}: {path}"
        )
    return decoded_frames, fps, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "ultimate_cpu.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = ROOT / config["experiment"]["output_root"]
    checked = 0
    for name, relative_source in config["experiment"]["videos"].items():
        source = ROOT / relative_source
        source_properties = video_properties(source)
        for method in config["experiment"]["methods"]:
            run = outputs / name / method
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if metrics["frames"] != source_properties[0]:
                raise AssertionError(f"Incorrect metrics frame count: {run}")
            for filename in ("annotated.mp4", "annotated_h264.mp4"):
                path = run / filename
                properties = video_properties(path)
                if properties[0] != source_properties[0]:
                    raise AssertionError(f"Truncated video: {path}")
                if abs(properties[1] - source_properties[1]) > 0.01:
                    raise AssertionError(f"FPS mismatch: {path}")
                if properties[2:] != source_properties[2:]:
                    raise AssertionError(f"Resolution mismatch: {path}")
            checked += 1
        print(f"PASS {name}: {source_properties[0]} frames, {len(config['experiment']['methods'])} methods")

    gallery = ROOT / "outputs" / "reports" / config["experiment"].get(
        "gallery_filename", "video_gallery.html"
    )
    html = gallery.read_text(encoding="utf-8")
    for name in config["experiment"]["videos"]:
        for method in config["experiment"]["methods"]:
            if method == "deep_ocsort_osnet":
                continue  # Kept on disk for historical comparison, hidden from galleries.
            relative_path = f"../{outputs.name}/{name}/{method}/annotated_h264.mp4"
            if relative_path not in html:
                raise AssertionError(f"Gallery missing {relative_path}")
    print(f"PASS: {checked} complete annotated outputs and gallery links")


if __name__ == "__main__":
    main()
