"""Video annotations and reproducible diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
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
    for track in tracks:
        color = identity_color(track.global_id)
        p1 = (int(round(track.x1)), int(round(track.y1)))
        p2 = (int(round(track.x2)), int(round(track.y2)))
        cv2.rectangle(annotated, p1, p2, color, 2)
        label = f"ID {track.global_id}"
        cv2.putText(
            annotated,
            label,
            (p1[0], max(18, p1[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
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
        ax.bar(x - width / 2, [baseline[n] for n in rate_names], width, label="Baseline", color="#176B87")
        ax.bar(x + width / 2, [improved[n] for n in rate_names], width, label="Improved", color="#D59F32")
        ax.set(title="Paper-aligned tracking accuracy", ylabel="Score (%)", xticks=x, xticklabels=rate_names)
        ax.set_ylim(0, 100)
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
        ax.bar(x - width / 2, [baseline[n] for n in count_names], width, label="Baseline", color="#176B87")
        ax.bar(x + width / 2, [improved[n] for n in count_names], width, label="Improved", color="#D59F32")
        ax.set(title="Tracking errors (lower is better)", ylabel="Count", xticks=x, xticklabels=count_names)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        fig.tight_layout()
        path = output / "tracking_errors_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(path)
    return created

