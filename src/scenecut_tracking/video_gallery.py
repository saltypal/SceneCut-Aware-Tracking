"""Build a browser gallery for a configured set of tracking results."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Iterable


VIDEO_NAMES = ("football-1", "football-2", "football-3", "MI6", "Joker")
VIDEO_LABELS = {"Joker": "Final clip"}
METHODS = (
    ("ocsort", "YOLO11 + OC-SORT"),
    ("deep_ocsort", "YOLO11 + Deep OC-SORT"),
    ("deep_ocsort_scenecut", "YOLO11 + Deep OC-SORT + Scene Cut + OSNet ReID"),
)


def _result_card(
    report_dir: Path,
    experiment_root: Path,
    video_name: str,
    method_key: str,
    method_label: str,
    video_labels: dict[str, str],
) -> str:
    run_dir = experiment_root / video_name / method_key
    video_path = run_dir / "annotated_h264.mp4"
    metrics_path = run_dir / "metrics.json"
    display_name = video_labels.get(video_name, video_name)
    title = f"{display_name} — {method_label}"

    if not video_path.is_file():
        return f"""
        <article class="card pending">
          <div class="card-heading"><h3>{html.escape(title)}</h3><span class="status">Pending</span></div>
          <div class="placeholder">Video will appear after this experiment finishes.</div>
        </article>
        """

    relative_video = Path(os.path.relpath(video_path, report_dir))
    chips: list[str] = []
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("frames") is not None:
            chips.append(f"{int(metrics['frames']):,} frames")
        if metrics.get("processing_fps") is not None:
            chips.append(f"{float(metrics['processing_fps']):.2f} FPS")
        if metrics.get("unique_global_ids") is not None:
            chips.append(f"{int(metrics['unique_global_ids']):,} IDs")
        if metrics.get("detected_scene_cuts"):
            chips.append(f"{int(metrics['detected_scene_cuts']):,} cuts")
        if metrics.get("accepted_recoveries"):
            chips.append(f"{int(metrics['accepted_recoveries']):,} unverified ReID matches")
    chip_html = "".join(f"<span class='chip'>{html.escape(chip)}</span>" for chip in chips)

    return f"""
    <article class="card complete">
      <div class="card-heading"><h3>{html.escape(title)}</h3><span class="status">Complete</span></div>
      <video controls preload="metadata">
        <source src="{html.escape(relative_video.as_posix())}" type="video/mp4">
        Your browser does not support HTML5 video.
      </video>
      <div class="chips">{chip_html}</div>
    </article>
    """


def build_video_gallery(
    project_root: str | Path,
    *,
    video_names: Iterable[str] | None = None,
    method_keys: Iterable[str] | None = None,
    output_root: str | Path = "outputs/experiments",
    output_name: str = "video_gallery.html",
    title: str = "SceneCut Aware Tracking — Video Results",
    video_labels: dict[str, str] | None = None,
) -> Path:
    """Create a gallery without assuming a fixed video or method count."""
    root = Path(project_root).resolve()
    experiment_root = (root / output_root).resolve()
    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if Path(output_name).name != output_name:
        raise ValueError("Gallery output_name must be a filename inside outputs/reports")
    output_path = report_dir / output_name

    names = tuple(video_names) if video_names is not None else VIDEO_NAMES
    requested_methods = set(method_keys) if method_keys is not None else None
    methods = tuple(
        (key, label) for key, label in METHODS
        if requested_methods is None or key in requested_methods
    )
    if not names or not methods:
        raise ValueError("Gallery needs at least one video and one visible method")
    labels = video_labels if video_labels is not None else (VIDEO_LABELS if output_name == "video_gallery.html" else {})

    method_headers = "".join(f"<div class='method-header'>{html.escape(label)}</div>" for _, label in methods)
    rows: list[str] = []
    completed = 0
    for video_name in names:
        cards = []
        for method_key, method_label in methods:
            if (experiment_root / video_name / method_key / "annotated_h264.mp4").is_file():
                completed += 1
            cards.append(_result_card(report_dir, experiment_root, video_name, method_key, method_label, labels))
        rows.append(
            f"<section class='video-row'><h2>{html.escape(labels.get(video_name, video_name))}</h2>"
            f"<div class='method-grid'>{''.join(cards)}</div></section>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#64748b; --line:#dbe3ef; --blue:#2563eb; --bg:#f6f8fc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:28px 32px 22px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:2; }}
    header h1 {{ margin:0 0 8px; font-size:28px; }}
    header p {{ margin:0; color:var(--muted); }}
    .progress {{ display:inline-block; margin-top:12px; padding:6px 10px; border-radius:999px; background:#e8f0ff; color:#174ea6; font-weight:700; }}
    main {{ max-width:1800px; margin:0 auto; padding:24px 30px 50px; }}
    .method-grid, .method-headers {{ display:grid; grid-template-columns:repeat({len(methods)},minmax(0,1fr)); gap:16px; }}
    .method-headers {{ margin-left:0; margin-bottom:12px; }}
    .method-header {{ font-weight:800; text-align:center; color:#334155; }}
    .video-row {{ margin:0 0 30px; }}
    .video-row > h2 {{ margin:0 0 12px; font-size:21px; }}
    .card {{ min-width:0; background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 5px 18px rgba(15,23,42,.06); }}
    .card-heading {{ min-height:48px; display:flex; align-items:flex-start; justify-content:space-between; gap:10px; }}
    .card h3 {{ margin:0; font-size:15px; line-height:1.35; }}
    .status {{ flex:none; border-radius:999px; padding:4px 8px; font-size:12px; font-weight:800; }}
    .complete .status {{ color:#166534; background:#dcfce7; }}
    .pending .status {{ color:#92400e; background:#fef3c7; }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#0f172a; border-radius:9px; }}
    .placeholder {{ display:grid; place-items:center; aspect-ratio:16/9; border:1px dashed #a8b4c7; border-radius:9px; color:var(--muted); text-align:center; padding:20px; background:#f8fafc; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:11px; min-height:25px; }}
    .chip {{ border-radius:999px; background:#eef2f7; padding:5px 8px; font-size:12px; color:#334155; }}
    footer {{ color:var(--muted); text-align:center; padding:0 20px 30px; }}
    @media (max-width:1200px) {{ .method-grid, .method-headers {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:700px) {{ .method-grid, .method-headers {{ grid-template-columns:1fr; }} .method-headers {{ display:none; }} main {{ padding:20px 16px 40px; }} header {{ padding:22px 18px; position:static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{len(names)} source videos × {len(methods)} tracking methods. Each source shares one YOLO11 detection cache. ID counts and accepted ReID matches are not identity-accuracy scores.</p>
    <span class="progress">{completed} of {len(names) * len(methods)} videos complete</span>
  </header>
  <main>
    <div class="method-headers">{method_headers}</div>
    {''.join(rows)}
  </main>
  <footer>CPU-only inference · pretrained models · no training or fine-tuning</footer>
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")
    return output_path
