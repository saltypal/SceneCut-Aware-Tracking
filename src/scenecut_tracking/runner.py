"""End-to-end video runners for baseline and cut-aware modes."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import save_config
from .detection_cache import DetectionCache
from .identity_memory import IdentityMemory
from .ocsort_tracker import OCSortTracker
from .runtime import environment_metadata, write_json
from .scene_cut import create_scene_cut_detector
from .types import TrackRecord
from .visualization import draw_tracks, plot_run_diagnostics


TRACK_COLUMNS = [
    "frame",
    "local_id",
    "global_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_id",
    "detection_index",
    "tracker_generation",
]


def _track_records(
    frame_index: int,
    raw_tracks: np.ndarray,
    global_ids: list[int],
    generation: int,
) -> list[TrackRecord]:
    records: list[TrackRecord] = []
    for row, global_id in zip(raw_tracks, global_ids):
        records.append(
            TrackRecord(
                frame=frame_index,
                local_id=int(row[4]),
                global_id=int(global_id),
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                confidence=float(row[5]),
                class_id=int(row[6]),
                detection_index=int(row[7]),
                tracker_generation=generation,
            )
        )
    return records


def _save_tracks(records: list[TrackRecord], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "tracks.csv"
    mot_path = output_dir / "tracks_mot.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    with mot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow(record.to_mot_row())
    return csv_path, mot_path


def run_video(
    mode: str,
    video_path: str | Path,
    detection_cache_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    project_root: str | Path,
    embedder=None,
) -> dict[str, Any]:
    if mode not in {"baseline", "improved"}:
        raise ValueError("mode must be 'baseline' or 'improved'")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = DetectionCache.load(detection_cache_path)
    metadata = cache.validate_video(video_path)
    tracker = OCSortTracker(config["tracker"])

    scene_detector = None
    identity_memory = None
    if mode == "improved":
        scene_detector = create_scene_cut_detector(config["scene_cut"], metadata.fps)
        if embedder is None:
            from .reid import OSNetEmbedder

            embedder = OSNetEmbedder(config["reid"], project_root)
        identity_memory = IdentityMemory(
            similarity_threshold=config["reid"]["similarity_threshold"],
            recovery_window_frames=config["reid"]["recovery_window_frames"],
            gallery_size=config["reid"]["gallery_size"],
        )

    capture = cv2.VideoCapture(str(Path(video_path).resolve()))
    writer = cv2.VideoWriter(
        str(output / f"annotated_{mode}.mp4"),
        cv2.VideoWriter_fourcc(*str(config["output"]["codec"])),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the annotated output video")

    records: list[TrackRecord] = []
    events: list[dict[str, Any]] = []
    frame_index = 0
    cut_frames: list[int] = []
    processing_started = time.perf_counter()
    sample_target_count = int(config["output"].get("save_sample_frames", 0))
    sample_indices = set(
        np.linspace(0, max(0, metadata.frame_count - 1), sample_target_count, dtype=int).tolist()
    ) if sample_target_count else set()
    sample_dir = output / "sample_frames"

    while True:
        success, frame = capture.read()
        if not success:
            break
        cut_detected = False
        if mode == "improved":
            decision = scene_detector.update(frame_index, frame)
            if decision.is_cut:
                cut_detected = True
                cut_frames.append(frame_index)
                tracker.reset_motion_state()
                identity_memory.begin_cut(frame_index)
                events.append(
                    {
                        "event": "scene_cut",
                        "frame": frame_index,
                        "score": decision.score,
                        "backend": decision.backend,
                    }
                )

        detections = cache.for_frame(frame_index)
        raw_tracks = tracker.update(detections, frame)
        local_ids = [int(row[4]) for row in raw_tracks]
        if mode == "baseline":
            global_ids = local_ids
        else:
            track_boxes = raw_tracks[:, :4] if len(raw_tracks) else np.empty((0, 4), dtype=np.float32)
            embeddings = embedder.extract(frame, track_boxes)
            global_ids, recovery_decisions = identity_memory.assign(frame_index, local_ids, embeddings)
            events.extend({"event": "identity_assignment", **asdict(item)} for item in recovery_decisions)

        frame_records = _track_records(frame_index, raw_tracks, global_ids, tracker.generation)
        records.extend(frame_records)
        annotated = draw_tracks(frame, frame_records, cut=cut_detected)
        writer.write(annotated)
        if frame_index in sample_indices:
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / f"frame_{frame_index:06d}.jpg"), annotated)
        frame_index += 1

    elapsed = time.perf_counter() - processing_started
    capture.release()
    writer.release()
    if frame_index == 0:
        raise RuntimeError("No video frames were decoded")

    tracks_csv, mot_path = _save_tracks(records, output)
    with (output / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
    plot_run_diagnostics(tracks_csv, output)
    save_config(config, output / "effective_config.yaml")

    recovered_events = [event for event in events if event.get("decision") == "recovered"]
    summary = {
        "mode": mode,
        "frames": frame_index,
        "source_fps": metadata.fps,
        "processing_fps": frame_index / max(elapsed, 1e-12),
        "elapsed_seconds": elapsed,
        "track_rows": len(records),
        "unique_global_ids": len({record.global_id for record in records}),
        "detected_scene_cuts": len(cut_frames),
        "cut_frames": cut_frames,
        "accepted_recoveries": len(recovered_events),
        "average_recovery_similarity": (
            float(np.mean([event["similarity"] for event in recovered_events]))
            if recovered_events
            else None
        ),
        "annotated_video": str(output / f"annotated_{mode}.mp4"),
        "tracks_mot": str(mot_path),
    }
    write_json(summary, output / "metrics.json")
    write_json(
        {
            "environment": environment_metadata(),
            "video": metadata.to_dict(),
            "detection_cache": str(Path(detection_cache_path).resolve()),
        },
        output / "run_metadata.json",
    )
    return summary

