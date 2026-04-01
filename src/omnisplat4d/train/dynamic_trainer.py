"""
Phase 2 — Train: SWinGS sliding-window 4DGS dynamic subject trainer.

Implements the Sliding-Window Gaussian Splatting (SWinGS) architecture for
training 4D Gaussian primitives on the dynamic subject isolated by Phase 2
masking. The full video is processed in 30-frame windows with 5-frame overlap.

Memory management is the dominant concern here. Each window trains its own
Gaussian state from scratch (initialised from the prior window's canonical
checkpoint), and all GPU allocations are purged between windows via
flush_cuda_cache(). Without this discipline, the gradient graph accumulates
across the full video sequence and causes immediate OOM on 12GB hardware.

Window output (per temporal chunk):
    - canonical_base.spz      — FP16 Gaussian positions, rotations, scales,
                                 opacities, DC colors.
    - deformation_field.onnx  — Trained DeformationMLP for this window.
    - ac_color_predictor.onnx — Trained ACColorMLP for this window.

The temporal consistency loss uses the frozen prior window's rendered frames as
a photometric reference for the overlapping 5 frames, penalising drift between
window boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omnisplat4d.core.config import DynamicTrainConfig
from omnisplat4d.core.memory import flush_cuda_cache
from omnisplat4d.core.types import FrameBatch, GaussianCheckpoint

log = logging.getLogger(__name__)


def train_dynamic(
    frame_batches: list[FrameBatch],
    canonical_init: GaussianCheckpoint,
    cfg: DynamicTrainConfig,
    output_dir: Path,
) -> list[Path]:
    """
    Train the full SWinGS 4DGS pipeline across all temporal windows.

    The outer loop iterates over temporal windows derived from frame_batches.
    Each window trains a DeformationMLP and ACColorMLP from scratch using the
    prior window's canonical GaussianCheckpoint as the initialisation point.
    The overlap region (cfg.window_overlap frames) applies the temporal
    consistency loss to minimise boundary drift.

    VRAM flush discipline:
        - flush_cuda_cache() is called unconditionally between every window.
        - All tensors from the completed window must be garbage-collected before
          the flush (del statements on the training state variables).

    Args:
        frame_batches:   List of FrameBatch objects covering the full video.
                         The caller is responsible for building these with the
                         correct window_size and window_overlap from cfg.
        canonical_init:  Initial GaussianCheckpoint to seed Window 0
                         (typically loaded from the static background SPZ or
                         initialised from the isolated subject point cloud).
        cfg:             DynamicTrainConfig block.
        output_dir:      Root output directory. Each window writes to
                         output_dir/chunk_NNN_frames_X_Y/.

    Returns:
        List of chunk directory Paths, one per temporal window, each containing:
            canonical_base.spz, deformation_field.onnx, ac_color_predictor.onnx.
    """
    raise NotImplementedError(
        "train_dynamic: implement SWinGS sliding-window training loop. "
        "See docs/research/phase2_4DGS_dynamic.md for architecture details."
    )


def _train_single_window(
    window_frames: FrameBatch,
    canonical_init: GaussianCheckpoint,
    prior_window_canonical: GaussianCheckpoint | None,
    window_idx: int,
    cfg: DynamicTrainConfig,
    output_dir: Path,
) -> tuple[GaussianCheckpoint, Path]:
    """
    Train one temporal window of the SWinGS pipeline.

    Args:
        window_frames:          FrameBatch for this window (window_size frames).
        canonical_init:         Initial Gaussian state for this window.
        prior_window_canonical: Frozen canonical state from the previous window,
                                used as photometric reference for consistency loss.
                                None for Window 0.
        window_idx:             Zero-based window index for naming output dirs.
        cfg:                    DynamicTrainConfig.
        output_dir:             Root output directory.

    Returns:
        (trained_canonical, chunk_dir) — the trained checkpoint and the
        path to the directory containing the serialised outputs.
    """
    raise NotImplementedError(
        "_train_single_window: implement per-window training loop with "
        "DeformationMLP, ACColorMLP, entropy loss, and temporal consistency loss."
    )


def build_frame_batches(
    frame_dir: Path,
    cfg: DynamicTrainConfig,
    num_cameras: int,
) -> list[FrameBatch]:
    """
    Build the list of overlapping FrameBatch objects for the sliding-window loop.

    Frames are grouped into windows of cfg.window_size with cfg.window_overlap
    frames shared between consecutive windows.

    Args:
        frame_dir:   Directory containing per-camera frame subdirectories
                     (cam_00/ … cam_NN/) with isolated subject frames.
        cfg:         DynamicTrainConfig (window_size, window_overlap).
        num_cameras: Number of virtual cameras.

    Returns:
        List of FrameBatch objects, where each batch covers one temporal window.
    """
    raise NotImplementedError("build_frame_batches: implement windowed batch construction.")
