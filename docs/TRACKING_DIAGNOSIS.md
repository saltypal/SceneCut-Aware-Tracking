# Tracking diagnosis and CPU rerun — 24 September 2026

## Scope and evidence

The active project is `D:\Bunker\BaseCamp\ComputerVision\SceneCut Aware tracking`.
The five complete inputs are `football-1`, `football-2`, `football-3`, `MI6`,
and `Joker`. All four methods use the same YOLO11n person-detection cache per
video: OC-SORT; standard Deep OC-SORT with OSNet; Deep OC-SORT with OSNet
appearance weight 1.0; and Deep OC-SORT with cut-aware identity memory.
**There is no separate DeepSORT implementation here.** No model was trained;
all inference ran on CPU.

The annotated SportsMOT football sequence
`data/SportsMOT-Example/dataset/train/v_gQNyhv8y0QY_c013` supplies identity
ground truth for calibration. The five custom clips do not have identity
ground truth. Their created-ID counts, cuts, and appearance matches are
diagnostics, **not** measured ID switches or tracking accuracy.

## Root causes

1. **Noisy detections started many short tracks.** YOLO cached person
   proposals down to confidence 0.05, and the old tracker also admitted new
   tracks at 0.05. In `football-3`, 41.2% of 13,947 proposals were below 0.20
   confidence and 21% were shorter than 48 pixels. On annotated SportsMOT,
   old OC-SORT produced 3,564 false positives and 421 true ID switches.
   YOLO11n is a small generic COCO person detector: tiny or blurred players,
   occlusion, and dramatic film lighting can still yield missing or imperfect
   boxes. A tracker cannot reconstruct a missed person detection.
   An 88-frame annotated detector check found that raising inference size
   from 640 to 960 gave only a small recall gain and worse precision, so
   resolution alone was not a justified fix for this one-day CPU setup.

2. **Appearance is still overlap-gated.** OC-SORT uses motion and IoU to
   associate boxes. Installed BoxMOT Deep OC-SORT adds OSNet cosine
   similarity, but `boxmot/utils/association.py` zeroes appearance weight
   where IoU is zero and rejects matched pairs below configured IoU 0.30.
   The direct-OSNet column merely sets appearance weight to 1.0 *within Deep
   OC-SORT*; it does not eliminate the spatial gate. Fast movement, a camera
   jump, occlusion, or a detector gap can therefore produce a new ID while
   the person remains in the shot. `max_age: 30` retains an internal track
   hypothesis, not a guaranteed visible labeled box. `min_hits: 3` can also
   delay displaying a newly detected person.

3. **False scene cuts reset motion unnecessarily.** PySceneDetect content
   score alone fired repeatedly during continuous Joker and football shots.
   Panning, flashes, crowds, and changing LED boards can create visual
   spikes. Each false reset broke short-term motion continuity and invoked
   risky cross-cut matching when no cut had happened.

4. **OSNet similarity is not identity proof.** Pretrained embeddings of
   similar team uniforms can look alike, while one person's embedding may
   shift with pose, lighting, scale, or partial occlusion. In sampled
   same-frame non-overlapping, reasonably sized detections, distinct people
   had cosine similarity at least 0.75 in 12.8%, 25.7%, and 25.4% of pairs
   for the three football clips. This is a threshold-risk diagnostic, *not*
   a measured false-recovery rate. Current memory searches historical
   prototypes for only 15 frames after an accepted cut. It does not fix
   ordinary same-shot losses, and a wrong match can contaminate a prototype.

5. **Project relocation hid config mistakes.** The old virtual environment's
   editable pointer still targeted `D:\Bunker\BaseCamp\SceneCut Aware tracking`.
   The single-video CLI used a legacy `default.yaml` with invalid comments,
   a bad model suffix, and uncalibrated values. The five-video runner and
   notebook actually used `configs/ultimate_cpu.yaml`, so edits to the old
   default did not affect them. The editable install, CLI default, and paths
   in current reused metadata are now repaired.

The box rendering is clearer after adding outlined rectangles and filled
labels. An audit of all new OC-SORT/Deep OC-SORT track rows found median IoU
1.0 against their assigned cached YOLO boxes, with zero rows below 0.5. Thus
the tracker renders the detector geometry faithfully; this does **not** mean
YOLO's box matches the real person's extent. Some visible people still lack
boxes because detection or tracker confirmation failed.
Visual inspection of the new football and Joker sample frames confirms that
some misses and partial-person boxes remain after calibration.

## Changes made

- Keep the shared detector cache at confidence 0.05, but require 0.25 to
  start new tracker hypotheses and allow the 0.10–0.25 range for existing
  OC-SORT tracks using BYTE association. The deep wrapper also supports a
  per-method threshold override for controlled tests.
- Require a PySceneDetect candidate plus global HSV histogram distance ≥0.40
  and normalized pixel difference ≥0.12 before declaring a hard cut. Space
  **accepted** cuts apart, and record visual evidence in event logs. This
  practical guard can still miss visually similar cuts or accept extreme
  within-shot transitions.
- Keep the designed separation between short-term tracker reset and
  long-term identity memory. Leave the 0.75 ReID threshold unchanged until
  labeled cross-cut pairs can validate it. Call accepted matches
  *unverified decisions* rather than correct recoveries.
- Repair the editable install and canonical configuration, refresh reused
  paths, and preserve old output under
  `outputs/archive_previous/before_2026-09-24_diagnosis`.

## Annotated football calibration: measured identity metrics

All six rows use the same YOLO and OSNet caches on one **875-frame SportsMOT
training sequence**. HOTA and IDF1 are higher when better; ID switches and
false positives are lower when better. Because this sequence helped choose
settings, these are exploratory calibration numbers, not independent test
performance.

| Method | Setting | HOTA ↑ | IDF1 ↑ | ID switches ↓ | FP ↓ |
|---|---|---:|---:|---:|---:|
| OC-SORT | old: entry 0.05, BYTE off | 33.66 | 29.22 | 421 | 3,564 |
| OC-SORT | tuned: entry 0.25, BYTE on | 43.66 | 42.83 | 262 | 2,218 |
| Deep OC-SORT | old | 38.74 | 33.50 | 381 | 3,716 |
| Deep OC-SORT | tuned | 50.52 | 50.39 | 169 | 1,318 |
| Deep OC-SORT, OSNet weight 1.0 | old | 38.78 | 35.23 | 354 | 3,758 |
| Deep OC-SORT, OSNet weight 1.0 | tuned | 51.25 | 51.13 | 150 | 1,317 |

The shared YOLO cache itself scored AP50 87.31, precision 58.74, recall
97.03 under this project's matching definition: high recall, but many false
proposals at 0.05. Full DetA, AssA, MOTA, FN, fragments, package versions,
and provenance are in `outputs/benchmarks/sportsmot_calibration/results.json`.
Its tracker-processing FPS excludes separately cached YOLO/OSNet inference,
so it is not end-to-end live-video FPS.

## Five custom clips: behavioral diagnostics only

Each ID cell shows **old → new created global IDs**. Fewer IDs can suggest
less fragmentation/noise but can also mean missed people. MI6's detection
cache was rebuilt during relocation, so its before/after row is not an
isolated tracker-only ablation.

| Video | OC-SORT IDs | Deep IDs | Deep + OSNet weight 1.0 IDs | Cut-aware IDs | Cut events | Accepted ReID decisions |
|---|---:|---:|---:|---:|---:|---:|
| football-1 | 216 → 85 | 209 → 74 | 212 → 74 | 226 → 77 | 11 → 8 | 27 → 3 |
| football-2 | 194 → 81 | 181 → 58 | 185 → 60 | 195 → 61 | 18 → 4 | 110 → 6 |
| football-3 | 457 → 165 | 395 → 151 | 388 → 156 | 394 → 138 | 6 → 4 | 39 → 13 |
| MI6 | 110 → 62 | 101 → 57 | 102 → 57 | 89 → 44 | 9 → 8 | 26 → 17 |
| Joker | 54 → 20 | 50 → 16 | 51 → 16 | 52 → 14 | 11 → 4 | 19 → 3 |

The cut guard reduced resets from 55 to 28 across all five videos and
accepted ReID decisions from 221 to 42. Many old decisions followed false
cuts, but **42 is not a correct-recovery count**. A `football-2` LED-board
transition may still pass the guard. Manual cut and identity annotations on
representative segments are the next evidence-based step; blind threshold
tuning would not establish reliability.

## Reproduce and verify

From the active project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\run_all_experiments.py
.\.venv\Scripts\python.exe scripts\validate_outputs.py
.\.venv\Scripts\python.exe scripts\benchmark_sportsmot.py
```

The default runner resumes validated caches and outputs. Add `--force` only
to rebuild costly CPU detection and appearance caches. Watch all 20 outputs
in `outputs/reports/video_gallery.html`; figures and tables belong in the
ultimate notebook. True custom-clip IDF1, switch counts, and cross-cut ReID
accuracy require identity-labeled frames. Until then, a guarantee that a
visible person's ID is retained across all shots or occlusions is unsupported.
The latest run manifest records the final no-change smoke test as `reused`;
the complete outputs were rebuilt in the preceding full CPU rerun.
