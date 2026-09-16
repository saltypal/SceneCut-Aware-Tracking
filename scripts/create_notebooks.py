"""Generate the two reader-facing experiment notebooks with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(cells):
    document = nbf.v4.new_notebook()
    document.cells = cells
    document.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    document.metadata.language_info = {"name": "python", "version": "3.13"}
    return document


def baseline_notebook():
    return notebook(
        [
            markdown(
                """
# Part 1 — YOLO + OC-SORT Baseline

This notebook runs the complete baseline: pretrained YOLO person detections are cached once and then passed to OC-SORT. It contains **inference and evaluation only—no training or fine-tuning**.

The bundled two-scene demo is a reproducible smoke test. Replace `VIDEO_PATH` and `GROUND_TRUTH_PATH` with football data for a real experiment.
"""
            ),
            markdown(
                """
## Goal and acceptance checks

- Detect only people (COCO class `0`).
- Save detections once for reuse by both experiments.
- Produce an annotated MP4 and MOT-format tracks.
- Calculate the OC-SORT paper metric family with official TrackEval when identity ground truth exists.
- Save readable diagnostic plots and exact run metadata.
"""
            ),
            code(
                """
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "configs" / "default.yaml").exists():
    raise RuntimeError("Start this notebook from the project root or notebooks directory.")

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from IPython.display import Image, Video, display
import pandas as pd

from scenecut_tracking.config import load_config
from scenecut_tracking.demo_data import create_demo_video
from scenecut_tracking.detection_cache import DetectionCache, build_detection_cache
from scenecut_tracking.detector import YoloPersonDetector
from scenecut_tracking.evaluation import evaluate_mot
from scenecut_tracking.runner import run_video
from scenecut_tracking.runtime import configure_runtime, write_json

configure_runtime(PROJECT_ROOT)
CONFIG = load_config(PROJECT_ROOT / "configs" / "default.yaml")
print(f"Project: {PROJECT_ROOT}")
"""
            ),
            markdown(
                """
## Data configuration

`USE_BUNDLED_DEMO=True` creates a short annotated video from real person crops with a hard cut at frame 24. This verifies the full pipeline. It is not a football benchmark.
"""
            ),
            code(
                """
USE_BUNDLED_DEMO = True
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "demo_people.mp4"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "annotations" / "demo_people_gt.txt"
DETECTION_CACHE_PATH = PROJECT_ROOT / "outputs" / "detections" / "demo_people.npz"
BASELINE_OUTPUT = PROJECT_ROOT / "outputs" / "baseline" / "demo"

detector = YoloPersonDetector(CONFIG["detector"], PROJECT_ROOT)
if USE_BUNDLED_DEMO:
    demo_metadata = create_demo_video(VIDEO_PATH, GROUND_TRUTH_PATH, detector)
    display(pd.DataFrame([demo_metadata]))
elif not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Provide a video at {VIDEO_PATH}")
"""
            ),
            markdown("## Run YOLO once and freeze its detections"),
            code(
                """
detection_cache = build_detection_cache(VIDEO_PATH, detector, CONFIG["detector"])
detection_cache.save(DETECTION_CACHE_PATH)
reloaded_cache = DetectionCache.load(DETECTION_CACHE_PATH)
reloaded_cache.validate_video(VIDEO_PATH)

assert detection_cache.rows.shape == reloaded_cache.rows.shape
assert (detection_cache.rows == reloaded_cache.rows).all()
print(f"Cached {len(reloaded_cache.rows):,} person detections at {DETECTION_CACHE_PATH}")
"""
            ),
            markdown("## Run the OC-SORT baseline"),
            code(
                """
baseline_summary = run_video(
    mode="baseline",
    video_path=VIDEO_PATH,
    detection_cache_path=DETECTION_CACHE_PATH,
    output_dir=BASELINE_OUTPUT,
    config=CONFIG,
    project_root=PROJECT_ROOT,
)
display(pd.DataFrame([baseline_summary]))
"""
            ),
            markdown(
                """
## Paper-aligned evaluation

These are the TrackEval implementations used for modern MOT evaluation. HOTA, AssA, IDF1, and MOTA are percentages; FP, FN, IDs, and Frag are counts. If a custom video has no identity annotations, these metrics must remain unavailable rather than being invented.
"""
            ),
            code(
                """
if GROUND_TRUTH_PATH.exists():
    paper_metrics = evaluate_mot(
        GROUND_TRUTH_PATH,
        BASELINE_OUTPUT / "tracks_mot.txt",
        iou_threshold=CONFIG["evaluation"]["match_iou_threshold"],
    )
    write_json(paper_metrics, BASELINE_OUTPUT / "paper_metrics.json")
    metric_order = ["HOTA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDs", "Frag"]
    display(pd.DataFrame({"Metric": metric_order, "Baseline": [paper_metrics[name] for name in metric_order]}))
else:
    paper_metrics = None
    print("Ground truth not supplied: paper metrics were not computed.")
"""
            ),
            markdown("## Inspect the baseline artifacts"),
            code(
                """
for plot_name in ["active_identities_over_time.png", "track_duration_distribution.png"]:
    plot_path = BASELINE_OUTPUT / plot_name
    if plot_path.exists():
        display(Image(filename=str(plot_path)))

display(Video(str(BASELINE_OUTPUT / "annotated_baseline.mp4"), embed=True, width=720))
"""
            ),
            markdown(
                """
## Checks and interpretation

The detection-cache equality assertion proves that downstream experiments can consume the same detections. The plots describe tracker behavior; only the TrackEval table measures accuracy. Demo results validate execution, not football-domain performance.

Next: run `02_SceneCut_ReID_Recovery.ipynb`, which keeps this cache fixed and changes only cut handling and identity recovery.
"""
            ),
        ]
    )


def improved_notebook():
    return notebook(
        [
            markdown(
                """
# Part 2 — Scene-Cut-Aware OC-SORT + OSNet Recovery

This notebook consumes the exact YOLO cache created in Part 1. At a hard cut it discards OC-SORT/Kalman motion state, preserves long-term OSNet appearance memory, and recovers an old identity only when conservative one-to-one matching passes the threshold.

There is **no training or fine-tuning**.
"""
            ),
            markdown(
                """
## Improvement invariant

```text
hard cut
├── discard: tracklets, Kalman state, local IDs
└── preserve: global IDs and bounded OSNet embedding galleries
```

The baseline and improved runs share the same `.npz` detection cache.
"""
            ),
            code(
                """
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "configs" / "default.yaml").exists():
    raise RuntimeError("Start this notebook from the project root or notebooks directory.")

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from IPython.display import Image, Video, display
import pandas as pd

from scenecut_tracking.config import load_config
from scenecut_tracking.demo_data import create_demo_video
from scenecut_tracking.detection_cache import DetectionCache, build_detection_cache
from scenecut_tracking.detector import YoloPersonDetector
from scenecut_tracking.evaluation import cross_cut_recovery_summary, evaluate_mot
from scenecut_tracking.runner import run_video
from scenecut_tracking.runtime import configure_runtime, write_json
from scenecut_tracking.visualization import plot_metric_comparison

configure_runtime(PROJECT_ROOT)
CONFIG = load_config(PROJECT_ROOT / "configs" / "default.yaml")
"""
            ),
            markdown("## Load the Part 1 inputs"),
            code(
                """
USE_BUNDLED_DEMO = True
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "demo_people.mp4"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "annotations" / "demo_people_gt.txt"
DETECTION_CACHE_PATH = PROJECT_ROOT / "outputs" / "detections" / "demo_people.npz"
BASELINE_OUTPUT = PROJECT_ROOT / "outputs" / "baseline" / "demo"
IMPROVED_OUTPUT = PROJECT_ROOT / "outputs" / "improved" / "demo"
CUT_FRAMES = [24] if USE_BUNDLED_DEMO else []

if not VIDEO_PATH.exists() or not DETECTION_CACHE_PATH.exists():
    detector = YoloPersonDetector(CONFIG["detector"], PROJECT_ROOT)
    if USE_BUNDLED_DEMO:
        create_demo_video(VIDEO_PATH, GROUND_TRUTH_PATH, detector)
    cache = build_detection_cache(VIDEO_PATH, detector, CONFIG["detector"])
    cache.save(DETECTION_CACHE_PATH)

shared_cache = DetectionCache.load(DETECTION_CACHE_PATH)
shared_cache.validate_video(VIDEO_PATH)
print(f"Using the frozen Part 1 cache with {len(shared_cache.rows):,} detections")
"""
            ),
            markdown("## Run cut-aware tracking and selective ReID"),
            code(
                """
improved_summary = run_video(
    mode="improved",
    video_path=VIDEO_PATH,
    detection_cache_path=DETECTION_CACHE_PATH,
    output_dir=IMPROVED_OUTPUT,
    config=CONFIG,
    project_root=PROJECT_ROOT,
)
display(pd.DataFrame([improved_summary]))
"""
            ),
            markdown("## Evaluate the same paper metrics"),
            code(
                """
if GROUND_TRUTH_PATH.exists():
    improved_metrics = evaluate_mot(
        GROUND_TRUTH_PATH,
        IMPROVED_OUTPUT / "tracks_mot.txt",
        iou_threshold=CONFIG["evaluation"]["match_iou_threshold"],
    )
    improved_metrics.update(
        cross_cut_recovery_summary(
            GROUND_TRUTH_PATH,
            IMPROVED_OUTPUT / "tracks_mot.txt",
            CUT_FRAMES,
            iou_threshold=CONFIG["evaluation"]["match_iou_threshold"],
        )
    )
    write_json(improved_metrics, IMPROVED_OUTPUT / "paper_metrics.json")
    display(pd.DataFrame([improved_metrics]))
else:
    improved_metrics = None
    print("Ground truth not supplied: paper and verified recovery metrics were not computed.")
"""
            ),
            markdown("## Baseline versus improved"),
            code(
                """
baseline_metrics_path = BASELINE_OUTPUT / "paper_metrics.json"
improved_metrics_path = IMPROVED_OUTPUT / "paper_metrics.json"
if baseline_metrics_path.exists() and improved_metrics_path.exists():
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    metric_order = ["HOTA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDs", "Frag"]
    comparison = pd.DataFrame({
        "Metric": metric_order,
        "Baseline": [baseline_metrics[name] for name in metric_order],
        "Improved": [improved_metrics[name] for name in metric_order],
    })
    comparison["Difference"] = comparison["Improved"] - comparison["Baseline"]
    display(comparison)
    plot_metric_comparison(baseline_metrics_path, improved_metrics_path, IMPROVED_OUTPUT)
else:
    print("Run Part 1 with ground truth before producing the comparison table.")
"""
            ),
            markdown("## Inspect plots, events, and video"),
            code(
                """
for plot_name in [
    "active_identities_over_time.png",
    "track_duration_distribution.png",
    "paper_metrics_comparison.png",
    "tracking_errors_comparison.png",
]:
    plot_path = IMPROVED_OUTPUT / plot_name
    if plot_path.exists():
        display(Image(filename=str(plot_path)))

events_path = IMPROVED_OUTPUT / "events.jsonl"
events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
display(pd.DataFrame(events).head(20))
display(Video(str(IMPROVED_OUTPUT / "annotated_improved.mp4"), embed=True, width=720))
"""
            ),
            markdown(
                """
## Takeaways

Use the comparison table—not the visual impression alone—to decide whether cut-aware recovery helped. On football footage, identical kits can make person ReID ambiguous; the conservative threshold deliberately favors a new ID over a false merge. Report demo findings only as pipeline validation until football identity annotations are available.
"""
            ),
        ]
    )


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(baseline_notebook(), NOTEBOOK_DIR / "01_YOLO_OCSORT_Baseline.ipynb")
    nbf.write(improved_notebook(), NOTEBOOK_DIR / "02_SceneCut_ReID_Recovery.ipynb")
    print("Created both notebooks")


if __name__ == "__main__":
    main()

