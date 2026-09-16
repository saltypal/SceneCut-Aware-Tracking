"""Command-line interface for reproducible detection, tracking, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runtime import configure_runtime, write_json


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs" / "default.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo", help="Create the annotated two-scene smoke-test video")
    demo.add_argument("--video", default=str(project_root() / "data" / "videos" / "demo_people.mp4"))
    demo.add_argument("--ground-truth", default=str(project_root() / "data" / "annotations" / "demo_people_gt.txt"))

    detect = subparsers.add_parser("detect", help="Run YOLO once and save the immutable detection cache")
    detect.add_argument("--video", required=True)
    detect.add_argument("--output", required=True)

    run = subparsers.add_parser("run", help="Run baseline or improved tracking from cached detections")
    run.add_argument("--mode", choices=["baseline", "improved"], required=True)
    run.add_argument("--video", required=True)
    run.add_argument("--detections", required=True)
    run.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Compute paper-aligned TrackEval metrics")
    evaluate.add_argument("--ground-truth", required=True)
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--cut-frames", nargs="*", type=int, default=[])

    compare = subparsers.add_parser("compare", help="Plot baseline and improved paper metrics")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--improved", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    configure_runtime(root)
    config = load_config(args.config)

    if args.command == "make-demo":
        from .demo_data import create_demo_video
        from .detector import YoloPersonDetector

        detector = YoloPersonDetector(config["detector"], root)
        result = create_demo_video(args.video, args.ground_truth, detector)
    elif args.command == "detect":
        from .detection_cache import build_detection_cache
        from .detector import YoloPersonDetector

        detector = YoloPersonDetector(config["detector"], root)
        cache = build_detection_cache(args.video, detector, config["detector"])
        cache.save(args.output)
        result = {"cache": str(Path(args.output).resolve()), "detections": len(cache.rows)}
    elif args.command == "run":
        from .runner import run_video

        result = run_video(
            mode=args.mode,
            video_path=args.video,
            detection_cache_path=args.detections,
            output_dir=args.output,
            config=config,
            project_root=root,
        )
    elif args.command == "evaluate":
        from .evaluation import cross_cut_recovery_summary, evaluate_mot

        result = evaluate_mot(
            args.ground_truth,
            args.prediction,
            iou_threshold=config["evaluation"]["match_iou_threshold"],
        )
        if args.cut_frames:
            result.update(
                cross_cut_recovery_summary(
                    args.ground_truth,
                    args.prediction,
                    args.cut_frames,
                    iou_threshold=config["evaluation"]["match_iou_threshold"],
                )
            )
        write_json(result, args.output)
    else:
        from .visualization import plot_metric_comparison

        paths = plot_metric_comparison(args.baseline, args.improved, args.output)
        result = {"plots": [str(path) for path in paths]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

