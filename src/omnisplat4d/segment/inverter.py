"""
Phase 2 — Segment: Background annihilation for dynamic subject isolation.

In Phase 1 (Stick Route), the operator mask is used to REMOVE the operator from
the background training set — the masked region is blacked out.

In Phase 2, the logic is inverted: the background is blacked out and only the
dynamic subject remains. This forces the 4DGS optimizer to constrain Gaussian
primitives to the moving actor, because the photometric loss penalises any
Gaussian that tries to reconstruct a pure-black background pixel.

Usage flow:
    operator_mask = masker.mask_frame(...)           # Phase 1: mask the operator
    subject_mask  = invert_mask(operator_mask)       # Phase 2: flip to mask background
    subject_frame = apply_black_background(frame, subject_mask)
"""

from __future__ import annotations

import numpy as np


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Invert a binary mask: regions that were masked become unmasked and vice versa.

    In Phase 1, the mask covers the operator (white = operator region).
    After inversion, the mask covers everything except the operator,
    which in Phase 2 represents the dynamic subject foreground region.

    Args:
        mask: Binary mask (H, W) uint8, where 255 = masked / 0 = unmasked.

    Returns:
        Inverted binary mask of the same shape.
    """
    return np.bitwise_not(mask)


def apply_black_background(
    frame_bgr: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """
    Zero out all background pixels, leaving only the subject region.

    Multiplies the frame by a binary mask so that background pixels become
    pure black (0, 0, 0). The 4DGS optimizer's photometric loss drives any
    Gaussian that reconstructs the black background toward zero opacity,
    effectively auto-pruning floating background artifacts.

    Args:
        frame_bgr:    Planar BGR frame as uint8 numpy array (H, W, 3).
        subject_mask: Binary mask (H, W) uint8, where 255 = subject / 0 = background.

    Returns:
        Frame with background zeroed, same shape as frame_bgr (H, W, 3) uint8.
    """
    binary = (subject_mask > 0).astype(np.uint8)
    return frame_bgr * binary[:, :, np.newaxis]


def apply_operator_mask(
    frame_bgr: np.ndarray,
    operator_mask: np.ndarray,
) -> np.ndarray:
    """
    Zero out the operator region, leaving only the clean background.

    This is the Phase 1 masking operation: the operator and selfie stick are
    removed from the frame before passing it to the 3DGS background trainer.

    Args:
        frame_bgr:     Planar BGR frame as uint8 numpy array (H, W, 3).
        operator_mask: Binary mask (H, W) uint8, where 255 = operator region.

    Returns:
        Frame with operator region zeroed, same shape as frame_bgr.
    """
    clean_mask = invert_mask(operator_mask)
    return apply_black_background(frame_bgr, clean_mask)
