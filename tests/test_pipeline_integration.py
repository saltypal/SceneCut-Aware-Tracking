from pathlib import Path

import cv2
import numpy as np

from scenecut_tracking.detection_cache import DetectionCache
from scenecut_tracking.runner import run_video
from scenecut_tracking.runtime import read_video_metadata


class ColorEmbedder:
    def extract(self, frame, boxes):
        features = []
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            mean_color = frame[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.float32)
            features.append(mean_color / max(np.linalg.norm(mean_color), 1e-12))
        return np.asarray(features, dtype=np.float32).reshape(-1, 3)


def test_improved_pipeline_resets_motion_and_recovers_ids(tmp_path: Path):
    video_path = tmp_path / "cut.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    boxes_by_frame = []
    for frame_index in range(6):
        after_cut = frame_index >= 3
        frame = np.full((120, 160, 3), (20, 80, 20) if not after_cut else (90, 50, 40), dtype=np.uint8)
        boxes = [(15 + frame_index, 20, 45 + frame_index, 90), (105 - frame_index, 25, 135 - frame_index, 95)]
        cv2.rectangle(frame, boxes[0][:2], boxes[0][2:], (20, 20, 230), -1)
        cv2.rectangle(frame, boxes[1][:2], boxes[1][2:], (230, 20, 20), -1)
        writer.write(frame)
        boxes_by_frame.append(boxes)
    writer.release()

    metadata = read_video_metadata(video_path)
    rows = []
    for frame_index, boxes in enumerate(boxes_by_frame):
        for box in boxes:
            rows.append([frame_index, *box, 0.99, 0])
    cache_path = tmp_path / "detections.npz"
    DetectionCache(
        np.asarray(rows, dtype=np.float32),
        {"schema_version": 1, "video": metadata.to_dict(), "detector": {"model": "fixture"}},
    ).save(cache_path)

    config = {
        "tracker": {
            "detection_threshold": 0.25,
            "max_age": 10,
            "min_hits": 1,
            "iou_threshold": 0.2,
            "delta_t": 2,
            "inertia": 0.2,
            "use_byte": False,
        },
        "scene_cut": {
            "backend": "pyscenedetect",
            "content_threshold": 10.0,
            "min_scene_length": 1,
            "histogram_threshold": 0.4,
        },
        "reid": {
            "similarity_threshold": 0.8,
            "recovery_window_frames": 2,
            "gallery_size": 4,
        },
        "output": {"codec": "mp4v", "save_sample_frames": 0},
    }
    output_dir = tmp_path / "output"
    summary = run_video(
        mode="improved",
        video_path=video_path,
        detection_cache_path=cache_path,
        output_dir=output_dir,
        config=config,
        project_root=tmp_path,
        embedder=ColorEmbedder(),
    )

    assert summary["detected_scene_cuts"] == 1
    assert summary["accepted_recoveries"] == 2
    assert summary["unique_global_ids"] == 2
    assert (output_dir / "annotated_improved.mp4").is_file()
    assert (output_dir / "tracks_mot.txt").is_file()
    assert (output_dir / "active_identities_over_time.png").is_file()

