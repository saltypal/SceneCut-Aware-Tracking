"""Full-frame Deep OC-SORT and cut-aware Deep OC-SORT video runners."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import save_config
from .appearance_cache import AppearanceCache
from .deep_ocsort_tracker import DeepOCSortTracker
from .detection_cache import DetectionCache
from .frame_source import iter_source_frames
from .identity_memory import IdentityMemory
from .runner import _save_tracks, _track_records
from .runtime import environment_metadata, write_json
from .scene_cut import create_scene_cut_detector
from .video_export import make_browser_video
from .visualization import draw_tracks, plot_run_diagnostics


VALID_MODES = {"deep_ocsort", "deep_ocsort_osnet", "deep_ocsort_scenecut"}


def run_deep_video(
    mode: str,
    video_path: str | Path,
    detection_cache_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    project_root: str | Path,
    appearance_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one Deep OC-SORT method over every source frame."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = DetectionCache.load(detection_cache_path)
    metadata = cache.validate_source(video_path)
    deep_config = dict(config["deep_ocsort"])
    if mode == "deep_ocsort_osnet":
        # Appearance-assisted OC-SORT column: use OSNet embeddings directly in
        # association, without scene-cut reset or long-term identity memory.
        deep_config["w_association_emb"] = 1.0
        deep_config["aw_off"] = True
    tracker = DeepOCSortTracker(config["tracker"], deep_config, project_root)
    appearance_cache = None
    if appearance_cache_path is not None:
        appearance_cache = AppearanceCache.load(appearance_cache_path)
        appearance_cache.validate(cache)

    cut_aware = mode == "deep_ocsort_scenecut"
    scene_detector = create_scene_cut_detector(config["scene_cut"], metadata.fps) if cut_aware else None
    identity_memory = (
        IdentityMemory(
            similarity_threshold=config["reid"]["similarity_threshold"],
            recovery_window_frames=config["reid"]["recovery_window_frames"],
            gallery_size=config["reid"]["gallery_size"],
        )
        if cut_aware
        else None
    )

    annotated_path = output / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*str(config["output"]["codec"])),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {annotated_path}")

    records = []
    events: list[dict[str, Any]] = []
    cut_frames: list[int] = []
    processed_frames = 0
    sample_target_count = int(config["output"].get("save_sample_frames", 0))
    sample_indices = (
        set(np.linspace(0, max(0, metadata.frame_count - 1), sample_target_count, dtype=int).tolist())
        if sample_target_count
        else set()
    )
    sample_dir = output / "sample_frames"
    started = time.perf_counter()

    try:
        for frame_index, frame in iter_source_frames(video_path):
            cut_detected = False
            if cut_aware:
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
                            "histogram_distance": decision.histogram_distance,
                            "pixel_difference": decision.pixel_difference,
                        }
                    )

            detections = cache.for_frame(frame_index)
            cached_embeddings = (
                appearance_cache.for_frame(frame_index)
                if appearance_cache is not None
                else None
            )
            raw_tracks, track_embeddings = tracker.update(
                detections, frame, detection_embeddings=cached_embeddings
            )
            local_ids = [int(row[4]) for row in raw_tracks]

            if cut_aware:
                global_ids, recovery_decisions = identity_memory.assign(
                    frame_index, local_ids, track_embeddings
                )
                events.extend(
                    {"event": "identity_assignment", **asdict(item)}
                    for item in recovery_decisions
                )
            else:
                global_ids = local_ids

            frame_records = _track_records(
                frame_index, raw_tracks, global_ids, tracker.generation
            )
            records.extend(frame_records)
            writer.write(draw_tracks(frame, frame_records, cut=cut_detected))
            if frame_index in sample_indices:
                sample_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sample_dir / f"frame_{frame_index:06d}.jpg"), draw_tracks(frame, frame_records, cut=cut_detected))
            processed_frames = frame_index + 1
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    if processed_frames != metadata.frame_count:
        raise RuntimeError(
            f"Processed {processed_frames} frames but source contains {metadata.frame_count}"
        )

    tracks_csv, mot_path = _save_tracks(records, output)
    with (output / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    plot_run_diagnostics(tracks_csv, output)
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_name in ("active_identities_over_time.png", "track_duration_distribution.png"):
        generated = output / plot_name
        if generated.exists():
            generated.replace(plots_dir / plot_name)

    effective_config = dict(config)
    effective_config["deep_ocsort"] = deep_config
    save_config(effective_config, output / "effective_config.yaml")
    browser_path = make_browser_video(annotated_path, output / "annotated_h264.mp4")
    recovered_events = [event for event in events if event.get("decision") == "recovered"]
    summary = {
        "mode": mode,
        "method": (
            "YOLO11 + Deep OC-SORT + Cut ReID"
            if cut_aware
            else "YOLO11 + Deep OC-SORT (OSNet weight 1.0)"
            if mode == "deep_ocsort_osnet"
            else "YOLO11 + Deep OC-SORT"
        ),
        "frames": processed_frames,
        "source_fps": metadata.fps,
        "processing_fps": processed_frames / max(elapsed, 1e-12),
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
        "annotated_video": str(annotated_path),
        "browser_video": str(browser_path),
        "tracks_mot": str(mot_path),
    }
    write_json(summary, output / "metrics.json")
    write_json(
        {
            "environment": environment_metadata(),
            "video": metadata.to_dict(),
            "detection_cache": str(Path(detection_cache_path).resolve()),
            "appearance_cache": (
                str(Path(appearance_cache_path).resolve())
                if appearance_cache_path is not None
                else None
            ),
            "reid_weights": str(
                Path(project_root).resolve()
                / "weights"
                / str(config["deep_ocsort"]["weights"])
            ),
        },
        output / "run_metadata.json",
    )
    return summary
