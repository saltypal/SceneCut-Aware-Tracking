# SceneCut Aware Tracking

CPU-only, no-training comparison of three multi-object tracking pipelines:

1. pretrained YOLO11 person detections + OC-SORT;
2. the same YOLO11 detections + Deep OC-SORT using pretrained OSNet;
3. the same YOLO11 detections + Deep OC-SORT + hard-scene-cut reset and
   conservative cross-cut identity recovery.

## Inputs

Every frame is processed for all four videos:

```text
data/videos/desktop_clips/football-1.mp4
data/videos/desktop_clips/football-2.mp4
data/videos/desktop_clips/football-3.mp4
data/MI6.mp4
```

The canonical experiment configuration is `configs/ultimate_cpu.yaml`.

## Controlled comparison

YOLO11 is run exactly once per source video. The resulting immutable detection
cache is shared by all three trackers for that video. Therefore tracker results
cannot differ because of different detector calls.

At a hard cut, the cut-aware method discards short-term motion/Kalman state but
preserves bounded long-term appearance memory. A previous global identity is
restored only when one-to-one OSNet similarity passes the configured threshold;
otherwise a new global ID is created.

## Outputs

All twelve experiment results live under `outputs/experiments`. See
`outputs/README.md` for the artifact contract. Previous results are preserved
under `outputs/archive_previous`.

## Notebook

The final presentation artifact will be:

```text
notebooks/Ultimate_YOLO11_Tracking_Comparison.ipynb
```

It will contain the architecture, full-frame validation, twelve result videos,
metrics tables, plots, event analysis, limitations, and reproduction commands.

## Evaluation honesty

Runtime, frame coverage, detected cuts, tracks, global IDs, ReID decisions, and
FPS can be reported for all videos. HOTA, AssA, IDF1, MOTA, true identity
switches, and true recovery accuracy require identity ground truth and will not
be fabricated for unannotated custom clips.
