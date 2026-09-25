"""Reproduce the old-vs-tuned CPU tracker check on the annotated SportsMOT example.

This is parameter calibration on one *training* sequence, not a held-out benchmark.
No model is trained. All six variants consume one shared YOLO11 detection cache.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scenecut_tracking.appearance_cache import AppearanceCache, build_appearance_cache
from scenecut_tracking.config import load_config
from scenecut_tracking.deep_ocsort_tracker import DeepOCSortTracker
from scenecut_tracking.detection_cache import DetectionCache, build_detection_cache
from scenecut_tracking.detector import YoloPersonDetector
from scenecut_tracking.evaluation import evaluate_detection_cache, evaluate_mot
from scenecut_tracking.frame_source import iter_source_frames
from scenecut_tracking.ocsort_tracker import OCSortTracker
from scenecut_tracking.runtime import environment_metadata, write_json


SEQUENCE = (
    PROJECT_ROOT / "data" / "SportsMOT-Example" / "dataset" / "train"
    / "v_gQNyhv8y0QY_c013"
)
OUTPUT = PROJECT_ROOT / "outputs" / "benchmarks" / "sportsmot_calibration"


def get_caches(config: dict) -> tuple[DetectionCache, AppearanceCache]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detection_path = OUTPUT / "detections.npz"
    appearance_path = OUTPUT / "embeddings.npz"
    if detection_path.is_file():
        detections = DetectionCache.load(detection_path)
        detections.validate_source(SEQUENCE)
        if detections.metadata["detector"] != config["detector"]:
            raise ValueError("Existing detection cache uses different YOLO settings; move it aside first")
        print("[cache] Shared YOLO detections: reuse", flush=True)
    else:
        print("[cache] Shared YOLO detections: build 875 CPU frames", flush=True)
        detector = YoloPersonDetector(config["detector"], PROJECT_ROOT)
        detections = build_detection_cache(SEQUENCE, detector, config["detector"])
        detections.save(detection_path)

    if appearance_path.is_file():
        appearance = AppearanceCache.load(appearance_path)
        appearance.validate(detections)
        print("[cache] Shared OSNet embeddings: reuse", flush=True)
    else:
        print("[cache] Shared OSNet embeddings: build from cached boxes", flush=True)
        appearance = build_appearance_cache(
            SEQUENCE, detection_path, config["reid"], PROJECT_ROOT
        )
        appearance.save(appearance_path)
    return detections, appearance


def run_variant(
    name: str,
    tracker_config: dict,
    deep_config: dict,
    detections: DetectionCache,
    appearance: AppearanceCache,
) -> dict:
    if name.startswith("ocsort"):
        tracker = OCSortTracker(tracker_config)
    else:
        tracker = DeepOCSortTracker(tracker_config, deep_config, PROJECT_ROOT)

    prediction_path = OUTPUT / f"{name}_mot.txt"
    started = time.perf_counter()
    row_count = 0
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for frame_index, frame in iter_source_frames(SEQUENCE):
            frame_detections = detections.for_frame(frame_index)
            if name.startswith("ocsort"):
                tracks = tracker.update(frame_detections, frame)
            else:
                tracks, _ = tracker.update(
                    frame_detections,
                    frame,
                    detection_embeddings=appearance.for_frame(frame_index),
                )
            for x1, y1, x2, y2, identity, *_ in tracks:
                writer.writerow((frame_index + 1, int(identity), float(x1), float(y1),
                                 float(x2 - x1), float(y2 - y1), 1, -1, -1, -1))
                row_count += 1

    scores = evaluate_mot(SEQUENCE / "gt" / "gt.txt", prediction_path)
    scores["tracker_processing_fps"] = round(
        detections.metadata["video"]["frame_count"] / max(time.perf_counter() - started, 1e-12), 2
    )
    scores["tracker_rows"] = row_count
    print(f"[result] {name}: HOTA={scores['HOTA']:.2f}, IDF1={scores['IDF1']:.2f}, "
          f"ID switches={scores['IDs']}", flush=True)
    return scores


def main() -> None:
    if not SEQUENCE.is_dir():
        raise FileNotFoundError(f"Annotated football sequence missing: {SEQUENCE}")
    config = load_config(PROJECT_ROOT / "configs" / "ultimate_cpu.yaml")
    if config["detector"]["device"] != "cpu" or config["reid"]["device"] != "cpu":
        raise ValueError("This calibration script is CPU-only")
    detections, appearance = get_caches(config)
    old_tracker = dict(config["tracker"], detection_threshold=0.05, use_byte=False)
    tuned_tracker = dict(config["tracker"], detection_threshold=0.25, use_byte=True)
    deep_standard = dict(config["deep_ocsort"])
    deep_direct = dict(deep_standard, w_association_emb=1.0, aw_off=True)
    variants = {
        "ocsort_old": (old_tracker, deep_standard),
        "ocsort_tuned": (tuned_tracker, deep_standard),
        "deep_ocsort_old": (old_tracker, deep_standard),
        "deep_ocsort_tuned": (tuned_tracker, deep_standard),
        "deep_direct_old": (old_tracker, deep_direct),
        "deep_direct_tuned": (tuned_tracker, deep_direct),
    }
    results = {
        name: run_variant(name, tracker_config, deep_config, detections, appearance)
        for name, (tracker_config, deep_config) in variants.items()
    }
    report = {
        "scope": "SportsMOT training example; exploratory calibration, not independent test",
        "sequence": str(SEQUENCE),
        "ground_truth": str(SEQUENCE / "gt" / "gt.txt"),
        "detector": evaluate_detection_cache(
            SEQUENCE / "gt" / "gt.txt", OUTPUT / "detections.npz"
        ),
        "variants": results,
        "environment": environment_metadata(),
    }
    write_json(report, OUTPUT / "results.json")
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDs", "Frag", "tracker_processing_fps")
        writer = csv.writer(handle)
        writer.writerow(("variant", *keys))
        for name, scores in results.items():
            writer.writerow((name, *(scores[key] for key in keys)))
    print(f"[saved] {OUTPUT / 'results.json'}")


if __name__ == "__main__":
    main()
