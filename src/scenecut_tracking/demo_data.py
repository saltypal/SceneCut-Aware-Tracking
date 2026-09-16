"""Create a small, annotated person-tracking video for deterministic smoke testing."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import cv2
import numpy as np


BUS_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def _download_bus_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(BUS_IMAGE_URL, path)


def create_demo_video(
    video_path: str | Path,
    ground_truth_path: str | Path,
    detector,
    frames_per_scene: int = 24,
    fps: float = 12.0,
) -> dict:
    """Build two hard-cut scenes using real person crops and known MOT identities."""
    video_path = Path(video_path)
    ground_truth_path = Path(ground_truth_path)
    source_path = video_path.parent / "bus_source.jpg"
    _download_bus_image(source_path)
    source = cv2.imread(str(source_path))
    if source is None:
        raise RuntimeError(f"Could not read downloaded image: {source_path}")
    detections = detector.detect(source)
    if len(detections) < 2:
        raise RuntimeError("YOLO did not find at least two people in the demo source image")
    person_rows = detections[np.argsort(-((detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])))[0:2]]
    crops: list[np.ndarray] = []
    for row in person_rows:
        x1, y1, x2, y2 = row[:4].astype(int)
        crop = source[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        crops.append(cv2.resize(crop, (72, 168), interpolation=cv2.INTER_AREA))

    width, height = 640, 360
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {video_path}")
    mot_rows: list[list[float]] = []
    total_frames = frames_per_scene * 2
    for frame_index in range(total_frames):
        second_scene = frame_index >= frames_per_scene
        local_frame = frame_index % frames_per_scene
        background_color = (42, 118, 42) if not second_scene else (66, 82, 128)
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)
        cv2.line(frame, (0, height // 2), (width, height // 2), (235, 235, 235), 2)
        positions = [
            (70 + 5 * local_frame, 112),
            (470 - 4 * local_frame, 118),
        ]
        if second_scene:
            positions = [
                (410 - 3 * local_frame, 96),
                (120 + 4 * local_frame, 126),
            ]
        for identity_index, (crop, (x, y)) in enumerate(zip(crops, positions), start=1):
            crop_height, crop_width = crop.shape[:2]
            frame[y:y + crop_height, x:x + crop_width] = crop
            mot_rows.append(
                [frame_index + 1, identity_index, x, y, crop_width, crop_height, 1, 1, 1]
            )
        writer.write(frame)
    writer.release()
    with ground_truth_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(mot_rows)
    return {
        "video": str(video_path.resolve()),
        "ground_truth": str(ground_truth_path.resolve()),
        "frames": total_frames,
        "hard_cut_frame_zero_based": frames_per_scene,
    }

