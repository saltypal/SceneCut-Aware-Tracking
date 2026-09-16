"""Runtime helpers shared by command-line and notebook entry points."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

from .types import VideoMetadata


def configure_runtime(project_root: str | Path) -> Path:
    """Keep third-party caches inside the project, never in a hidden user directory."""
    runtime_root = Path(project_root).resolve() / ".runtime"
    ultralytics_dir = runtime_root / "ultralytics"
    torch_dir = runtime_root / "torch"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    return runtime_root


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file size plus first/last chunks without loading a large video into memory."""
    source = Path(path).resolve()
    size = source.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with source.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(0, size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def read_video_metadata(path: str | Path) -> VideoMetadata:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    metadata = VideoMetadata(
        path=str(source),
        fingerprint=file_fingerprint(source),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    if metadata.frame_count <= 0 or metadata.width <= 0 or metadata.height <= 0:
        raise ValueError(f"Video metadata is invalid: {metadata}")
    return metadata


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package_name in ["numpy", "opencv-python", "torch", "ultralytics", "boxmot", "scenedetect", "trackeval"]:
        try:
            from importlib.metadata import version

            packages[package_name] = version(package_name)
        except Exception:
            packages[package_name] = "unavailable"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "uncommitted-or-not-a-git-repository"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
    }


def write_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

