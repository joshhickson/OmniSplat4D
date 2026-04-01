"""
Phase 1/2 — Segment: SAM 2.1 Tiny pixel-perfect segmentation.

Takes YOLOv8 bounding boxes as prompts and produces binary segmentation masks.
Critical memory management for 12GB VRAM hardware:

    - Vision features are offloaded to CPU RAM (sam_storage_device="cpu").
    - The vision feature cache is capped at 1 entry (sam_max_vision_cache=1).
    - flush_cuda_cache() is called after EVERY frame inference.
    - reset_tracker() is called whenever IoU drops below iou_reset_threshold.

Violating any of these constraints will exhaust VRAM within the masking loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from omnisplat4d.core.config import SegmentConfig
from omnisplat4d.core.memory import flush_cuda_cache

log = logging.getLogger(__name__)


def load_sam(cfg: SegmentConfig) -> object:
    """
    Load the SAM 2.1 Tiny video predictor with CPU offloading configured.

    Key settings applied:
        - storage_device = cfg.sam_storage_device (default "cpu")
          Routes temporal state and vision features to host RAM.
        - max_vision_features_cache_size = cfg.sam_max_vision_cache (default 1)
          Strictly limits in-memory feature caches to prevent OOM.

    Args:
        cfg: SegmentConfig block.

    Returns:
        A SAM2VideoPredictor instance ready for frame-by-frame inference.

    Raises:
        ImportError: If sam-2 is not installed (install with extras: [segment]).
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise ImportError(
            "sam-2 is required for masking. Install with: pip install 'omnisplat4d[segment]'"
        ) from e

    log.info(
        "Loading SAM 2.1 predictor: checkpoint=%s, device=%s, storage=%s",
        cfg.sam_checkpoint, cfg.sam_inference_device, cfg.sam_storage_device,
    )
    predictor = build_sam2_video_predictor(
        config_file=cfg.sam_config,
        ckpt_path=cfg.sam_checkpoint,
        device=cfg.sam_inference_device,
    )
    # Apply memory constraints
    if hasattr(predictor, "max_vision_features_cache_size"):
        predictor.max_vision_features_cache_size = cfg.sam_max_vision_cache
    if hasattr(predictor, "storage_device"):
        predictor.storage_device = cfg.sam_storage_device

    return predictor


def mask_frame(
    predictor: object,
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    cfg: SegmentConfig,
) -> np.ndarray:
    """
    Generate a binary segmentation mask for one frame given a bounding box prompt.

    Calls flush_cuda_cache() unconditionally after inference regardless of success
    or failure. This is non-negotiable on 12GB VRAM hardware.

    Args:
        predictor: Loaded SAM 2.1 predictor from load_sam().
        frame_bgr: Planar BGR frame as a uint8 numpy array (H, W, 3).
        bbox:      Bounding box prompt as [x1, y1, x2, y2] float32.
        cfg:       SegmentConfig (used for device routing).

    Returns:
        Binary mask of shape (H, W) uint8, where 255 = masked region (operator/subject).
    """
    import torch

    try:
        # SAM 2.1 expects RGB
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            predictor.set_image(frame_rgb)
            masks, scores, _ = predictor.predict(
                box=bbox[None],  # SAM expects (1, 4)
                multimask_output=False,
            )
        # masks shape: (1, H, W) bool
        mask = (masks[0].astype(np.uint8)) * 255
        return mask
    finally:
        flush_cuda_cache()


def reset_tracker(predictor: object) -> None:
    """
    Reset SAM 2.1 tracking state after a confidence drop.

    Called when IoU between consecutive frames drops below
    cfg.iou_reset_threshold, typically during occlusion or re-entry after
    the subject exits and re-enters the camera frustum.

    Args:
        predictor: Loaded SAM 2.1 predictor from load_sam().
    """
    if hasattr(predictor, "reset_tracking_data"):
        predictor.reset_tracking_data()
        log.debug("SAM 2.1 tracking state reset")
    flush_cuda_cache()


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    Used to detect tracking confidence drops that should trigger reset_tracker().

    Args:
        mask_a: Binary mask (H, W) uint8.
        mask_b: Binary mask (H, W) uint8.

    Returns:
        IoU as a float in [0.0, 1.0].
    """
    a = mask_a > 0
    b = mask_b > 0
    intersection = float((a & b).sum())
    union = float((a | b).sum())
    return intersection / union if union > 0 else 0.0
