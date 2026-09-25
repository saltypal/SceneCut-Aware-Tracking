# Unified experiment outputs

The active output tree contains five videos and four methods per video:

```text
outputs/
├── detection_cache/
├── appearance_cache/
├── experiments/
│   ├── football-1/
│   │   ├── ocsort/
│   │   ├── deep_ocsort/
│   │   ├── deep_ocsort_osnet/
│   │   └── deep_ocsort_scenecut/
│   ├── football-2/
│   ├── football-3/
│   ├── MI6/
│   └── Joker/
├── benchmarks/sportsmot_calibration/
├── comparisons/figures/
├── reports/
└── archive_previous/before_2026-09-24_diagnosis/
```

Every completed experiment directory must contain:

- `annotated.mp4`: original full-frame rendered result.
- `annotated_h264.mp4`: browser-compatible full-frame result.
- `tracks.csv`: readable track records.
- `tracks_mot.txt`: MOTChallenge-format track records.
- `metrics.json`: runtime and behavioral diagnostics.
- `events.jsonl`: scene-cut and recovery events, empty when not applicable.
- `effective_config.yaml`: exact configuration used.
- `run_metadata.json`: input and environment provenance.
- `plots/`: per-run diagnostic figures.

The detector is run once per source video. All four methods for a video must
consume the same immutable YOLO11 detection cache.

Run `python scripts/run_all_experiments.py` from the project root to resume or
recompute the complete experiment. It records configuration signatures so a
changed detector, tracker, ReID, or scene-cut parameter invalidates the correct
artifacts automatically.

Historical pre-diagnosis outputs were moved, not deleted. They remain under
`archive_previous/before_2026-09-24_diagnosis` with their original names and
metadata. The SportsMOT benchmark uses one annotated *training* sequence for
exploratory calibration, not for independent custom-clip accuracy claims.
