"""Video annotations and reproducible diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import TrackRecord


def identity_color(identity_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(identity_id * 104729)
    color = rng.integers(55, 235, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(frame: np.ndarray, tracks: list[TrackRecord], cut: bool = False) -> np.ndarray:
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    font_scale = max(0.45, min(0.70, frame_height / 1080.0 * 0.72))
    for track in tracks:
        color = identity_color(track.global_id)
        p1 = (
            int(np.clip(round(track.x1), 0, frame_width - 1)),
            int(np.clip(round(track.y1), 0, frame_height - 1)),
        )
        p2 = (
            int(np.clip(round(track.x2), 0, frame_width - 1)),
            int(np.clip(round(track.y2), 0, frame_height - 1)),
        )
        if p2[0] <= p1[0] or p2[1] <= p1[1]:
            continue
        # The dark outer stroke keeps boxes visible on both bright pitch and dark film shots.
        cv2.rectangle(annotated, p1, p2, (0, 0, 0), 4)
        cv2.rectangle(annotated, p1, p2, color, 2)
        label = f"ID {track.global_id}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_width = text_width + 10
        label_height = text_height + 8
        label_x = int(np.clip(p1[0], 0, max(0, frame_width - label_width)))
        label_y = p1[1] - label_height if p1[1] >= label_height else p1[1] + 1
        label_y = int(np.clip(label_y, 0, max(0, frame_height - label_height)))
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_x + label_width, label_y + label_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 5, label_y + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    if cut:
        cv2.putText(
            annotated,
            "HARD CUT: motion state reset",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (20, 20, 240),
            2,
            cv2.LINE_AA,
        )
    return annotated


def plot_run_diagnostics(tracks_csv: str | Path, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(tracks_csv)
    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if len(tracks):
        active = tracks.groupby("frame")["global_id"].nunique()
        full_index = np.arange(int(tracks["frame"].max()) + 1)
        active = active.reindex(full_index, fill_value=0)
        ax.plot(active.index, active.values, color="#176B87", linewidth=2)
    ax.set(title="Active tracked identities by frame", xlabel="Frame", ylabel="Active identities")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    fig.tight_layout()
    path = output / "active_identities_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if len(tracks):
        durations = tracks.groupby("global_id")["frame"].nunique()
        ax.hist(durations.values, bins=min(20, max(5, len(durations))), color="#D59F32", edgecolor="#30343B")
    ax.set(title="Track-duration distribution", xlabel="Observed frames per identity", ylabel="Identity count")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    fig.tight_layout()
    path = output / "track_duration_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(path)
    return created


def plot_metric_comparison(
    baseline_metrics: str | Path,
    improved_metrics: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    with Path(baseline_metrics).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with Path(improved_metrics).open("r", encoding="utf-8") as handle:
        improved = json.load(handle)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    rate_names = [name for name in ["HOTA", "AssA", "IDF1", "MOTA"] if name in baseline and name in improved]
    if rate_names:
        x = np.arange(len(rate_names))
        width = 0.36
        fig, ax = plt.subplots(figsize=(9, 5))
        baseline_values = [baseline[name] for name in rate_names]
        improved_values = [improved[name] for name in rate_names]
        baseline_bars = ax.bar(x - width / 2, baseline_values, width, label="Baseline", color="#176B87")
        improved_bars = ax.bar(x + width / 2, improved_values, width, label="Improved", color="#D59F32")
        ax.set(title="Paper-aligned tracking accuracy", ylabel="Score (%)", xticks=x, xticklabels=rate_names)
        all_values = baseline_values + improved_values
        lower = min(0.0, min(all_values))
        upper = max(0.0, max(all_values))
        span = max(upper - lower, 1.0)
        ax.set_ylim(lower - 0.12 * span, upper + 0.20 * span)
        ax.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(improved_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        fig.tight_layout()
        path = output / "paper_metrics_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)

    count_names = [name for name in ["FP", "FN", "IDs", "Frag"] if name in baseline and name in improved]
    if count_names:
        x = np.arange(len(count_names))
        width = 0.36
        fig, ax = plt.subplots(figsize=(9, 5))
        baseline_values = [baseline[name] for name in count_names]
        improved_values = [improved[name] for name in count_names]
        baseline_bars = ax.bar(x - width / 2, baseline_values, width, label="Baseline", color="#176B87")
        improved_bars = ax.bar(x + width / 2, improved_values, width, label="Improved", color="#D59F32")
        ax.set(title="Tracking errors (lower is better; symmetric log scale)", ylabel="Count", xticks=x, xticklabels=count_names)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.bar_label(baseline_bars, fmt="%.0f", padding=3, fontsize=8)
        ax.bar_label(improved_bars, fmt="%.0f", padding=3, fontsize=8)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        fig.tight_layout()
        path = output / "tracking_errors_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)
    return created
