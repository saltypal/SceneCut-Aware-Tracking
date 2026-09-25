"""Reproducibly run every configured detector, tracker, and recovery pass.

Run this file from anywhere with the project virtual environment::

    python scripts/run_all_experiments.py

The YAML file is the single source of truth.  Cache and result signatures make
normal reruns resumable: changing detector settings rebuilds detections, while
changing tracker/ReID/scene-cut settings rebuilds only the affected runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenecut_tracking.appearance_cache import AppearanceCache, build_appearance_cache
from scenecut_tracking.config import load_config, merged_config
from scenecut_tracking.detection_cache import DetectionCache, build_detection_cache
from scenecut_tracking.deep_runner import run_deep_video
from scenecut_tracking.detector import YoloPersonDetector
from scenecut_tracking.runner import run_video
from scenecut_tracking.runtime import configure_runtime, environment_metadata, read_video_metadata, write_json
from scenecut_tracking.video_export import make_browser_video
from scenecut_tracking.video_gallery import build_video_gallery


ALL_METHODS = ("ocsort", "deep_ocsort", "deep_ocsort_osnet", "deep_ocsort_scenecut")


def _canonical(value: Any) -> str:
    """Return a stable representation suitable for a configuration signature."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _signature(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _method_signature(config: dict[str, Any], method: str, detection_signature: str, appearance_signature: str | None) -> str:
    payload: dict[str, Any] = {
        "method": method,
        "detector_signature": detection_signature,
        "tracker": config["tracker"],
        "output": config["output"],
    }
    if method != "ocsort":
        payload["deep_ocsort"] = config["deep_ocsort"]
        payload["reid"] = config["reid"]
        payload["appearance_signature"] = appearance_signature
    if method == "deep_ocsort_scenecut":
        payload["scene_cut"] = config["scene_cut"]
    return _signature(payload)


def _safe_remove_method_dir(method_dir: Path, experiment_root: Path) -> None:
    """Remove only a known method directory below outputs/experiments."""
    resolved = method_dir.resolve()
    boundary = experiment_root.resolve()
    if boundary not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside experiment root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _cache_is_current(cache_path: Path, video_path: Path, detector_config: dict[str, Any], signature: str) -> bool:
    if not cache_path.is_file():
        return False
    try:
        cache = DetectionCache.load(cache_path)
        cache.validate_source(video_path)
        return (
            cache.metadata.get("pipeline_signature") == signature
            and cache.metadata.get("detector") == detector_config
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _get_detection_cache(
    video_name: str,
    video_path: Path,
    detector: YoloPersonDetector,
    config: dict[str, Any],
    cache_root: Path,
) -> tuple[Path, str, str]:
    detector_config = deepcopy(config["detector"])
    detector_signature = _signature(detector_config)
    cache_path = cache_root / f"{video_name}_yolo11n_cpu.npz"
    if _cache_is_current(cache_path, video_path, detector_config, detector_signature):
        cache = DetectionCache.load(cache_path)
        current_path = str(video_path.resolve())
        if cache.metadata["video"].get("path") != current_path:
            cache.metadata["video"]["path"] = current_path
            cache.save(cache_path)
        print(f"[cache] detections {video_name}: reuse")
        return cache_path, detector_signature, "reused"

    print(f"[cache] detections {video_name}: rebuild")
    cache = build_detection_cache(video_path, detector, detector_config)
    cache.metadata["pipeline_signature"] = detector_signature
    cache.save(cache_path)
    return cache_path, detector_signature, "rebuilt"


def _appearance_is_current(
    cache_path: Path,
    video_path: Path,
    detection_cache: DetectionCache,
    reid_config: dict[str, Any],
    signature: str,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        cache = AppearanceCache.load(cache_path)
        cache.validate(detection_cache)
        return cache.metadata.get("pipeline_signature") == signature
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _get_appearance_cache(
    video_name: str,
    video_path: Path,
    detection_path: Path,
    detection_signature: str,
    config: dict[str, Any],
    appearance_root: Path,
) -> tuple[Path, str, str]:
    reid_signature = _signature({"reid": config["reid"], "detection_signature": detection_signature})
    appearance_path = appearance_root / f"{video_name}_osnet_x0_25_msmt17_cpu.npz"
    detection_cache = DetectionCache.load(detection_path)
    if _appearance_is_current(appearance_path, video_path, detection_cache, config["reid"], reid_signature):
        appearance = AppearanceCache.load(appearance_path)
        current_path = str(video_path.resolve())
        current_weights = str((PROJECT_ROOT / "weights" / config["reid"]["weights"]).resolve())
        if (appearance.metadata["video"].get("path") != current_path
                or appearance.metadata["reid"].get("weights") != current_weights):
            appearance.metadata["video"]["path"] = current_path
            appearance.metadata["reid"]["weights"] = current_weights
            appearance.save(appearance_path)
        print(f"[cache] OSNet embeddings {video_name}: reuse")
        return appearance_path, reid_signature, "reused"

    print(f"[cache] OSNet embeddings {video_name}: rebuild")
    appearance_cache = build_appearance_cache(
        video_path=video_path,
        detection_cache_path=detection_path,
        reid_config=config["reid"],
        project_root=PROJECT_ROOT,
    )
    appearance_cache.metadata["pipeline_signature"] = reid_signature
    appearance_cache.save(appearance_path)
    return appearance_path, reid_signature, "rebuilt"


def _output_is_current(output_dir: Path, signature: str, video_path: Path) -> bool:
    required = (output_dir / "metrics.json", output_dir / "annotated.mp4", output_dir / "annotated_h264.mp4", output_dir / "tracks.csv")
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
        source = read_video_metadata(video_path)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        return (
            metadata.get("pipeline_signature") == signature
            and metrics.get("frames") == source.frame_count
            and metadata.get("video", {}).get("fingerprint") == source.fingerprint
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _refresh_reused_output_paths(
    output_dir: Path,
    video_path: Path,
    detection_path: Path,
    appearance_path: Path | None,
) -> dict[str, Any]:
    """Keep resumable results portable when the whole project is moved."""
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_metadata.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics["annotated_video"] = str((output_dir / "annotated.mp4").resolve())
    metrics["browser_video"] = str((output_dir / "annotated_h264.mp4").resolve())
    metrics["tracks_mot"] = str((output_dir / "tracks_mot.txt").resolve())
    metadata["video"] = read_video_metadata(video_path).to_dict()
    metadata["source_path"] = str(video_path.resolve())
    metadata["detection_cache"] = str(detection_path.resolve())
    metadata["appearance_cache"] = str(appearance_path.resolve()) if appearance_path else None
    write_json(metrics, metrics_path)
    write_json(metadata, metadata_path)
    return metrics


def _run_ocsort(
    video_name: str,
    video_path: Path,
    detection_path: Path,
    detection_signature: str,
    config: dict[str, Any],
    output_dir: Path,
    force: bool,
) -> tuple[dict[str, Any], str]:
    signature = _method_signature(config, "ocsort", detection_signature, None)
    if not force and _output_is_current(output_dir, signature, video_path):
        print(f"[run] {video_name}/ocsort: reuse")
        return _refresh_reused_output_paths(output_dir, video_path, detection_path, None), "reused"

    _safe_remove_method_dir(output_dir, output_dir.parent.parent)
    print(f"[run] {video_name}/ocsort: execute")
    metrics = run_video(
        mode="baseline",
        video_path=video_path,
        detection_cache_path=detection_path,
        output_dir=output_dir,
        config=config,
        project_root=PROJECT_ROOT,
    )
    generated = output_dir / "annotated_baseline.mp4"
    standard = output_dir / "annotated.mp4"
    if standard.exists():
        standard.unlink()
    generated.replace(standard)
    make_browser_video(standard, output_dir / "annotated_h264.mp4")
    metrics["mode"] = "ocsort"
    metrics["method"] = "YOLO11 + OC-SORT"
    metrics["annotated_video"] = str(standard)
    metrics["browser_video"] = str(output_dir / "annotated_h264.mp4")
    write_json(metrics, output_dir / "metrics.json")
    _write_signature_metadata(output_dir, video_path, detection_path, None, signature)
    return metrics, "rebuilt"


def _run_deep(
    video_name: str,
    method: str,
    video_path: Path,
    detection_path: Path,
    detection_signature: str,
    appearance_path: Path,
    appearance_signature: str,
    config: dict[str, Any],
    output_dir: Path,
    force: bool,
) -> tuple[dict[str, Any], str]:
    signature = _method_signature(config, method, detection_signature, appearance_signature)
    if not force and _output_is_current(output_dir, signature, video_path):
        print(f"[run] {video_name}/{method}: reuse")
        return _refresh_reused_output_paths(
            output_dir, video_path, detection_path, appearance_path
        ), "reused"

    _safe_remove_method_dir(output_dir, output_dir.parent.parent)
    print(f"[run] {video_name}/{method}: execute")
    metrics = run_deep_video(
        mode=method,
        video_path=video_path,
        detection_cache_path=detection_path,
        appearance_cache_path=appearance_path,
        output_dir=output_dir,
        config=config,
        project_root=PROJECT_ROOT,
    )
    _write_signature_metadata(output_dir, video_path, detection_path, appearance_path, signature)
    return metrics, "rebuilt"


def _write_signature_metadata(
    output_dir: Path,
    video_path: Path,
    detection_path: Path,
    appearance_path: Path | None,
    signature: str,
) -> None:
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["pipeline_signature"] = signature
    metadata["source_path"] = str(video_path)
    metadata["detection_cache"] = str(detection_path)
    metadata["appearance_cache"] = str(appearance_path) if appearance_path else None
    write_json(metadata, metadata_path)


def _iter_videos(config: dict[str, Any], requested: Iterable[str] | None) -> list[tuple[str, Path]]:
    configured = config["experiment"]["videos"]
    names = list(requested) if requested else list(configured)
    unknown = sorted(set(names) - set(configured))
    if unknown:
        raise ValueError(f"Unknown video name(s): {', '.join(unknown)}")
    return [(name, _resolve_path(PROJECT_ROOT, configured[name])) for name in names]


def execute(config_path: Path, force: bool = False, requested_videos: Iterable[str] | None = None, requested_methods: Iterable[str] | None = None) -> dict[str, Any]:
    configure_runtime(PROJECT_ROOT)
    config = load_config(config_path)
    videos = _iter_videos(config, requested_videos)
    overrides = config["experiment"].get("video_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("experiment.video_overrides must be a mapping")
    unknown_overrides = sorted(set(overrides) - set(config["experiment"]["videos"]))
    if unknown_overrides:
        raise ValueError(f"Overrides name unknown videos: {', '.join(unknown_overrides)}")
    allowed_sections = {"detector", "tracker", "deep_ocsort", "scene_cut", "reid", "output"}
    for video_name, override in overrides.items():
        if not isinstance(override, dict) or any(
            section not in allowed_sections or not isinstance(settings, dict)
            for section, settings in override.items()
        ):
            raise ValueError(f"Invalid pipeline override for video: {video_name}")
    configured_methods = tuple(config["experiment"].get("methods", ALL_METHODS))
    methods = tuple(requested_methods) if requested_methods else configured_methods
    invalid_methods = sorted(set(methods) - set(ALL_METHODS))
    if invalid_methods:
        raise ValueError(f"Unsupported method(s): {', '.join(invalid_methods)}")

    cache_root = _resolve_path(PROJECT_ROOT, config["experiment"]["detection_cache_root"])
    appearance_root = _resolve_path(PROJECT_ROOT, config["experiment"]["appearance_cache_root"])
    experiment_root = _resolve_path(PROJECT_ROOT, config["experiment"]["output_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    appearance_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "config": str(config_path.resolve()),
        "force": force,
        "environment": environment_metadata(),
        "runs": [],
    }
    for video_name, video_path in videos:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_config = merged_config(config, overrides.get(video_name, {}))
        detector = YoloPersonDetector(video_config["detector"], PROJECT_ROOT)
        detection_path, detection_signature, detection_status = _get_detection_cache(
            video_name, video_path, detector, video_config, cache_root
        )
        appearance_path: Path | None = None
        appearance_signature: str | None = None
        if any(method != "ocsort" for method in methods):
            appearance_path, appearance_signature, appearance_status = _get_appearance_cache(
                video_name, video_path, detection_path, detection_signature, video_config, appearance_root
            )
        else:
            appearance_status = "not requested"

        manifest["runs"].append({"video": video_name, "stage": "detection_cache", "status": detection_status})
        if appearance_path is not None:
            manifest["runs"].append({"video": video_name, "stage": "appearance_cache", "status": appearance_status})
        for method in methods:
            output_dir = experiment_root / video_name / method
            if method == "ocsort":
                metrics, status = _run_ocsort(
                    video_name, video_path, detection_path, detection_signature, video_config, output_dir, force
                )
            else:
                assert appearance_path is not None and appearance_signature is not None
                metrics, status = _run_deep(
                    video_name, method, video_path, detection_path, detection_signature,
                    appearance_path, appearance_signature, video_config, output_dir, force
                )
            manifest["runs"].append({"video": video_name, "method": method, "status": status, "frames": metrics.get("frames")})

    experiment_config = config["experiment"]
    gallery = build_video_gallery(
        PROJECT_ROOT,
        video_names=experiment_config["videos"],
        method_keys=methods,
        output_root=experiment_config["output_root"],
        output_name=experiment_config.get("gallery_filename", "video_gallery.html"),
        title=experiment_config.get("gallery_title", "SceneCut Aware Tracking — Video Results"),
        video_labels=experiment_config.get("video_labels"),
    )
    manifest["gallery"] = str(gallery)
    manifest_path = PROJECT_ROOT / "outputs" / "reports" / experiment_config.get(
        "manifest_filename", "last_run_manifest.json"
    )
    write_json(manifest, manifest_path)
    print(f"[gallery] {gallery}")
    print(f"[manifest] {manifest_path}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ultimate_cpu.yaml"))
    parser.add_argument("--force", action="store_true", help="Recompute requested caches and runs even when signatures match")
    parser.add_argument("--video", action="append", dest="videos", help="Restrict to a configured video; repeat for several")
    parser.add_argument("--method", action="append", dest="methods", choices=ALL_METHODS, help="Restrict to a method; repeat for several")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    execute(Path(args.config).resolve(), force=args.force, requested_videos=args.videos, requested_methods=args.methods)
