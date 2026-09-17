"""Create browser-compatible H.264 copies of generated annotated videos."""

from __future__ import annotations

import subprocess
from pathlib import Path


def make_browser_video(input_path: str | Path, output_path: str | Path) -> Path:
    """Transcode a generated MP4 to H.264/yuv420p with fast-start metadata."""
    import imageio_ffmpeg

    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {source}:\n{completed.stderr[-2000:]}")
    return output
