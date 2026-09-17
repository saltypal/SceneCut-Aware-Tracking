"""Read either a video file or one MOTChallenge image sequence directly."""

from __future__ import annotations

import configparser
import hashlib
from pathlib import Path
from typing import Iterator

import cv2

from .runtime import file_fingerprint
from .types import VideoMetadata


def _resolve_image_directory(source: Path) -> tuple[Path, Path | None]:
    if source.name.lower() == "img1":
        sequence_root = source.parent
        info_path = sequence_root / "seqinfo.ini"
        return source, info_path if info_path.is_file() else None
    image_directory = source / "img1"
    if image_directory.is_dir():
        info_path = source / "seqinfo.ini"
        return image_directory, info_path if info_path.is_file() else None
    return source, None


def _image_sequence_fingerprint(images: list[Path]) -> str:
    digest = hashlib.sha256()
    for image in (images[0], images[-1]):
        digest.update(image.name.encode("utf-8"))
        digest.update(str(image.stat().st_size).encode("ascii"))
        with image.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
    digest.update(str(len(images)).encode("ascii"))
    return digest.hexdigest()


def read_source_metadata(path: str | Path) -> VideoMetadata:
    """Read metadata from a video or directly from an MOT image directory."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        from .runtime import read_video_metadata

        return read_video_metadata(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Frame source not found: {source}")

    image_directory, info_path = _resolve_image_directory(source)
    images = sorted(image_directory.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG frames found in {image_directory}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Could not decode {images[0]}")
    height, width = first.shape[:2]
    fps = 25.0
    if info_path is not None:
        parser = configparser.ConfigParser()
        parser.read(info_path, encoding="utf-8")
        fps = parser.getfloat("Sequence", "frameRate", fallback=fps)
        expected = parser.getint("Sequence", "seqLength", fallback=len(images))
        if expected != len(images):
            raise ValueError(f"seqinfo.ini expects {expected} frames but found {len(images)}")
    return VideoMetadata(
        path=str(source),
        fingerprint=_image_sequence_fingerprint(images),
        frame_count=len(images),
        width=width,
        height=height,
        fps=fps,
    )


def iter_source_frames(path: str | Path) -> Iterator[tuple[int, object]]:
    """Yield zero-based frame indices and BGR frames in deterministic order."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {source}")
        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame_index, frame
                frame_index += 1
        finally:
            capture.release()
        return

    image_directory, _ = _resolve_image_directory(source)
    for frame_index, image_path in enumerate(sorted(image_directory.glob("*.jpg"))):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not decode {image_path}")
        yield frame_index, frame
