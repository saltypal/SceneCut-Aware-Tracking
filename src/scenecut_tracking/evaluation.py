"""Official TrackEval metric calculation from MOT-format ground truth and predictions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .detection_cache import DetectionCache


def _load_mot(path: str | Path, is_ground_truth: bool) -> dict[int, list[tuple[int, np.ndarray]]]:
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 6:
                raise ValueError(f"MOT row {line_number} has fewer than six columns: {path}")
            values = [float(value) for value in row]
            frame = int(values[0])
            identity = int(values[1])
            x, y, width, height = values[2:6]
            confidence = values[6] if len(values) > 6 else 1.0
            if is_ground_truth and confidence <= 0:
                continue
            box = np.asarray([x, y, x + width, y + height], dtype=np.float64)
            by_frame[frame].append((identity, box))
    return by_frame


def box_iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=np.float64)
    top_left = np.maximum(gt_boxes[:, None, :2], tracker_boxes[None, :, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], tracker_boxes[None, :, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    gt_area = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    tracker_area = np.maximum(0.0, tracker_boxes[:, 2] - tracker_boxes[:, 0]) * np.maximum(0.0, tracker_boxes[:, 3] - tracker_boxes[:, 1])
    union = gt_area[:, None] + tracker_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def _trackeval_data(ground_truth: dict, predictions: dict) -> dict:
    frames = sorted(set(ground_truth) | set(predictions))
    gt_id_values = sorted({identity for rows in ground_truth.values() for identity, _ in rows})
    tracker_id_values = sorted({identity for rows in predictions.values() for identity, _ in rows})
    gt_id_map = {identity: index for index, identity in enumerate(gt_id_values)}
    tracker_id_map = {identity: index for index, identity in enumerate(tracker_id_values)}

    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    for frame in frames:
        gt_rows = ground_truth.get(frame, [])
        tracker_rows = predictions.get(frame, [])
        gt_ids.append(np.asarray([gt_id_map[item[0]] for item in gt_rows], dtype=int))
        tracker_ids.append(np.asarray([tracker_id_map[item[0]] for item in tracker_rows], dtype=int))
        gt_boxes = np.asarray([item[1] for item in gt_rows], dtype=np.float64).reshape(-1, 4)
        tracker_boxes = np.asarray([item[1] for item in tracker_rows], dtype=np.float64).reshape(-1, 4)
        similarities.append(box_iou_matrix(gt_boxes, tracker_boxes))
        num_gt_dets += len(gt_rows)
        num_tracker_dets += len(tracker_rows)
    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_id_values),
        "num_tracker_ids": len(tracker_id_values),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def evaluate_mot(ground_truth_path: str | Path, prediction_path: str | Path, iou_threshold: float = 0.5) -> dict:
    """Compute the exact HOTA, CLEAR, and Identity implementations used by TrackEval."""
    # TrackEval currently uses removed NumPy aliases internally. This compatibility shim
    # does not change metric behavior.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    from trackeval.metrics import CLEAR, HOTA, Identity

    ground_truth = _load_mot(ground_truth_path, is_ground_truth=True)
    predictions = _load_mot(prediction_path, is_ground_truth=False)
    data = _trackeval_data(ground_truth, predictions)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota["HOTA"]) * 100.0),
        "DetA": float(np.mean(hota["DetA"]) * 100.0),
        "AssA": float(np.mean(hota["AssA"]) * 100.0),
        "IDF1": float(identity["IDF1"] * 100.0),
        "MOTA": float(clear["MOTA"] * 100.0),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "IDs": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "GT_detections": int(data["num_gt_dets"]),
        "Predicted_detections": int(data["num_tracker_dets"]),
    }


def cross_cut_recovery_summary(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    cut_frames: list[int],
    window: int = 10,
    iou_threshold: float = 0.5,
) -> dict:
    """Measure whether GT identities retain the same predicted ID around annotated cuts."""
    ground_truth = _load_mot(ground_truth_path, is_ground_truth=True)
    predictions = _load_mot(prediction_path, is_ground_truth=False)
    verified = 0
    recovered = 0
    for cut_frame_zero_based in cut_frames:
        cut_frame = cut_frame_zero_based + 1
        before_matches: dict[int, list[int]] = defaultdict(list)
        after_matches: dict[int, list[int]] = defaultdict(list)
        for frame, destination in [
            (value, before_matches) for value in range(max(1, cut_frame - window), cut_frame)
        ] + [
            (value, after_matches) for value in range(cut_frame, cut_frame + window)
        ]:
            gt_rows = ground_truth.get(frame, [])
            pred_rows = predictions.get(frame, [])
            if not gt_rows or not pred_rows:
                continue
            similarities = box_iou_matrix(
                np.asarray([row[1] for row in gt_rows]),
                np.asarray([row[1] for row in pred_rows]),
            )
            from scipy.optimize import linear_sum_assignment

            gt_indices, pred_indices = linear_sum_assignment(-similarities)
            for gt_index, pred_index in zip(gt_indices, pred_indices):
                if similarities[gt_index, pred_index] >= iou_threshold:
                    destination[gt_rows[gt_index][0]].append(pred_rows[pred_index][0])
        for gt_id in set(before_matches) & set(after_matches):
            before_id = Counter(before_matches[gt_id]).most_common(1)[0][0]
            after_id = Counter(after_matches[gt_id]).most_common(1)[0][0]
            verified += 1
            recovered += int(before_id == after_id)
    return {
        "cross_cut_identities_evaluated": verified,
        "cross_cut_identities_recovered": recovered,
        "cross_cut_recovery_rate": (100.0 * recovered / verified) if verified else None,
    }


def evaluate_detection_cache(
    ground_truth_path: str | Path,
    cache_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate cached person detections with one-to-one IoU matching and AP50."""
    ground_truth = _load_mot(ground_truth_path, is_ground_truth=True)
    cache = DetectionCache.load(cache_path)
    total_ground_truth = sum(len(rows) for rows in ground_truth.values())
    ranked: list[tuple[float, int, np.ndarray]] = []
    frame_count = int(cache.metadata["video"]["frame_count"])
    for frame_index in range(frame_count):
        for detection in cache.for_frame(frame_index):
            ranked.append((float(detection[4]), frame_index + 1, detection[:4].astype(float)))
    ranked.sort(key=lambda item: item[0], reverse=True)

    matched: dict[int, set[int]] = defaultdict(set)
    true_positive = np.zeros(len(ranked), dtype=float)
    false_positive = np.zeros(len(ranked), dtype=float)
    for detection_index, (_, frame, box) in enumerate(ranked):
        frame_ground_truth = ground_truth.get(frame, [])
        if not frame_ground_truth:
            false_positive[detection_index] = 1.0
            continue
        gt_boxes = np.asarray([row[1] for row in frame_ground_truth], dtype=float)
        overlaps = box_iou_matrix(gt_boxes, box.reshape(1, 4))[:, 0]
        candidate_order = np.argsort(-overlaps)
        match_index = next(
            (
                int(candidate)
                for candidate in candidate_order
                if overlaps[candidate] >= iou_threshold and int(candidate) not in matched[frame]
            ),
            None,
        )
        if match_index is None:
            false_positive[detection_index] = 1.0
        else:
            matched[frame].add(match_index)
            true_positive[detection_index] = 1.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall_curve = cumulative_tp / max(total_ground_truth, 1)
    precision_curve = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision_curve[recall_curve >= level]))
        if np.any(recall_curve >= level)
        else 0.0
        for level in recall_points
    ]
    average_precision = float(np.mean(interpolated))
    tp = int(cumulative_tp[-1]) if len(cumulative_tp) else 0
    fp = int(cumulative_fp[-1]) if len(cumulative_fp) else 0
    fn = int(total_ground_truth - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_ground_truth, 1)
    runtime = cache.metadata.get("runtime", {})
    return {
        "IoU_threshold": float(iou_threshold),
        "AP50": 100.0 * average_precision,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "F1": 100.0 * (2.0 * precision * recall / max(precision + recall, 1e-12)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "GT_detections": int(total_ground_truth),
        "Predicted_detections": int(len(ranked)),
        "average_detections_per_frame": len(ranked) / max(frame_count, 1),
        "processing_fps": runtime.get("processing_fps"),
        "elapsed_seconds": runtime.get("elapsed_seconds"),
    }
