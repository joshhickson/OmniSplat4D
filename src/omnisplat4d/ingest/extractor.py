"""
Phase 1 — Ingestion: Frame extraction and keyframe selection.

Responsibilities:
    1. Invoke FFmpeg (via subprocess) to decode the input equirectangular .mp4
       at the configured target FPS.
    2. Apply Laplacian variance sharpness scoring to each candidate frame.
    3. Retain only frames above the sharpness threshold, discarding blurry
       frames caused by motion blur during capture.
    4. Write accepted frames as numbered PNG files to workspace/extracted_frames/.

FFmpeg is invoked as a subprocess (not via ffmpeg-python) for explicit control
over the exact command string during debugging.

Memory contract: frames are decoded one at a time via a stdout pipe.
The full video is never held in RAM simultaneously.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from omnisplat4d.core.config import IngestConfig

log = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    cfg: IngestConfig,
) -> list[Path]:
    """
    Extract keyframes from an equirectangular video using FFmpeg + Laplacian filtering.

    Frames are decoded at cfg.fps_target and written to output_dir only if their
    Laplacian variance exceeds cfg.laplacian_threshold.

    Args:
        video_path: Absolute path to the source .mp4 file.
        output_dir: Directory where accepted keyframe PNGs are written.
        cfg:        IngestConfig block from PipelineConfig.

    Returns:
        Sorted list of absolute paths to accepted keyframe PNGs,
        named frame_XXXXXX.png (zero-padded 6 digits).

    Side effects:
        Creates output_dir (and parents) if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []
    for frame_idx, bgr in iter_ffmpeg_frames(video_path, cfg.fps_target):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = laplacian_variance(gray)
        if score < cfg.laplacian_threshold:
            log.debug("Frame %06d rejected (sharpness %.1f < %.1f)", frame_idx, score, cfg.laplacian_threshold)
            continue
        out_path = output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), bgr)
        accepted.append(out_path)
        log.debug("Frame %06d accepted (sharpness %.1f)", frame_idx, score)
    log.info("Extracted %d keyframes to %s", len(accepted), output_dir)
    return sorted(accepted)


def laplacian_variance(image: np.ndarray) -> float:
    """
    Compute the Laplacian variance of a grayscale image as a sharpness proxy.

    Higher values indicate sharper frames. Used to filter motion-blurred frames
    before SfM. Values below IngestConfig.laplacian_threshold are rejected.

    Args:
        image: Grayscale uint8 numpy array, shape (H, W).

    Returns:
        Scalar variance of the Laplacian response (float).
    """
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(lap.var())


def iter_ffmpeg_frames(
    video_path: Path,
    fps: float,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Generator that decodes video frames at a target FPS via FFmpeg stdout pipe.

    Spawns FFmpeg with the rawvideo output codec and reads one frame at a time
    from stdout. Only one frame resides in memory at any given time.

    Args:
        video_path: Path to the source video file.
        fps:        Target decode frame rate (e.g. 2.0 for 2 fps).

    Yields:
        (frame_index, bgr_frame) tuples where frame_index is sequential at
        the requested FPS rate and bgr_frame is a uint8 numpy array (H, W, 3).

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError:      If FFmpeg fails to start or probe the video dimensions.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Probe video dimensions via ffprobe
    width, height = _probe_dimensions(video_path)
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None

    frame_idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        yield frame_idx, frame.copy()
        frame_idx += 1

    proc.stdout.close()
    proc.wait()


def _probe_dimensions(video_path: Path) -> tuple[int, int]:
    """
    Use ffprobe to retrieve the (width, height) of a video file.

    Args:
        video_path: Path to the video.

    Returns:
        (width, height) as integers.

    Raises:
        RuntimeError: If ffprobe cannot parse the dimensions.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(",")
    if len(parts) != 2:
        raise RuntimeError(f"ffprobe returned unexpected output: {result.stdout!r}")
    return int(parts[0]), int(parts[1])
