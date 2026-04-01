"""
Phase 1/2 — Segment: YOLOv8-nano bounding box detection.

Provides lightweight, fast bounding box detection for the operator/selfie-stick
(Phase 1 Stick Route) or the dynamic subject (Phase 2). The bounding box is
then passed to masker.py as a prompt for SAM 2.1's pixel-perfect segmentation.

YOLOv8-nano is chosen because:
    - It runs on CPU without significant latency.
    - Its memory footprint leaves VRAM headroom for SAM 2.1 and training.
    - Class 'person' (COCO class 0) covers both operator and subject cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from omnisplat4d.core.config import SegmentConfig

log = logging.getLogger(__name__)


def load_detector(cfg: SegmentConfig) -> object:
    """
    Load the YOLOv8-nano model from cfg.yolo_model.

    The model is loaded onto the inference device specified by the YOLO
    library defaults (CPU inference is the norm for 12GB VRAM budget).

    Args:
        cfg: SegmentConfig block.

    Returns:
        A loaded ultralytics YOLO model instance.

    Raises:
        ImportError: If ultralytics is not installed (install with extras: [segment]).
        FileNotFoundError: If the weights file does not exist and cannot be downloaded.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for detection. Install with: pip install 'omnisplat4d[segment]'"
        ) from e

    log.info("Loading YOLOv8 detector from %s", cfg.yolo_model)
    model = YOLO(cfg.yolo_model)
    return model


def detect_operator(
    frame_bgr: np.ndarray,
    model: object,
    cfg: SegmentConfig,
) -> Optional[np.ndarray]:
    """
    Detect the human operator (and selfie stick) in a planar frame.

    Runs YOLOv8 inference and returns the highest-confidence 'person' bounding
    box. Returns None if no person is detected above the model's default threshold.

    Args:
        frame_bgr: Planar BGR frame as a uint8 numpy array (H, W, 3).
        model:     Loaded YOLO model from load_detector().
        cfg:       SegmentConfig (currently unused but kept for future thresholds).

    Returns:
        Bounding box as [x1, y1, x2, y2] float32 array in pixel coordinates,
        or None if no person detected.
    """
    results = model(frame_bgr, classes=[0], verbose=False)  # class 0 = person
    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    # Pick highest-confidence detection
    best_idx = int(boxes.conf.argmax())
    xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(np.float32)
    return xyxy  # [x1, y1, x2, y2]
