# SceneCut Aware Tracking

Minimal, modular person tracking with two controlled experiments:

1. **Baseline:** pretrained YOLO detections followed by OC-SORT.
2. **Improved:** the same detections followed by OC-SORT, hard-cut detection, motion-state reset, and conservative OSNet identity recovery.

There is no training or fine-tuning in this repository.

## Why detections are cached

YOLO runs once. Both experiments consume the same immutable `.npz` cache, so any difference comes from tracking and cross-cut identity handling rather than different detector results.

## Environment

BoxMOT supports Python 3.10–3.13. On this machine use Python 3.13:

```powershell
cd "D:\Bunker\BaseCamp\SceneCut Aware tracking"
$env:UV_CACHE_DIR = "$PWD\.runtime\uv-cache"
uv venv .venv --python C:\py\python.exe
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

## Notebooks

Run in order:

```text
notebooks/01_YOLO_OCSORT_Baseline.ipynb
notebooks/02_SceneCut_ReID_Recovery.ipynb
```

Each notebook can create a small two-scene validation video when no football video is supplied. That demo validates execution; it is not evidence of football-domain accuracy.

## Command-line reproduction

Create the small annotated demo:

```powershell
.venv\Scripts\python.exe -m scenecut_tracking.cli make-demo
```

Run YOLO once:

```powershell
.venv\Scripts\python.exe -m scenecut_tracking.cli detect `
  --video data\videos\demo_people.mp4 `
  --output outputs\detections\demo_people.npz
```

Run the baseline:

```powershell
.venv\Scripts\python.exe -m scenecut_tracking.cli run `
  --mode baseline `
  --video data\videos\demo_people.mp4 `
  --detections outputs\detections\demo_people.npz `
  --output outputs\baseline\demo
```

Run the improvement:

```powershell
.venv\Scripts\python.exe -m scenecut_tracking.cli run `
  --mode improved `
  --video data\videos\demo_people.mp4 `
  --detections outputs\detections\demo_people.npz `
  --output outputs\improved\demo
```

Evaluate with the official TrackEval implementations used by OC-SORT:

```powershell
.venv\Scripts\python.exe -m scenecut_tracking.cli evaluate `
  --ground-truth data\annotations\demo_people_gt.txt `
  --prediction outputs\baseline\demo\tracks_mot.txt `
  --output outputs\baseline\demo\paper_metrics.json
```

The paper-aligned table contains HOTA, AssA, IDF1, MOTA, FP, FN, identity switches (`IDs`), and fragmentations (`Frag`). Metrics require identity ground truth. The notebooks will not fabricate these values when a custom video has no annotations.

## Design invariant at a hard cut

```text
discard: OC-SORT tracks + Kalman motion state + local tracker-ID mapping
preserve: global identity memory + bounded OSNet embedding galleries
```

New-shot tracks are assigned an old global identity only after one-to-one cosine-similarity matching passes the configured threshold. Otherwise, they receive a new identity.

## Limitations

- Person ReID is difficult in football because teammates wear nearly identical kits.
- The conservative threshold intentionally favors missed recovery over false identity merging.
- Published OC-SORT scores used different detectors and benchmark datasets. This project uses the same evaluation metrics, not the same expected scores.
- Real football accuracy must be measured with football identity annotations.

