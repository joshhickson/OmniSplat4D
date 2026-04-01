"""
Phase 3 — Composite: Unified CUDA depth-sort renderer.

Solves the occlusion problem between static background and dynamic subject
Gaussians by concatenating both sets of primitives before rasterisation and
sorting them together by depth. This produces mathematically correct occlusion
without Z-buffer overhead.

Algorithm:
    1. Concatenate static + deformed dynamic Gaussians into one point cloud.
       Both sets share the same world coordinate system (from the same COLMAP
       camera poses), so concatenation is valid without any transformation.
    2. Project all primitives to screen space and compute orthogonal depth
       relative to the camera plane.
    3. Assign each Gaussian a 64-bit sort key:
           high 32 bits = tile ID  (screen divided into 16×16 px tiles)
           low 32 bits  = float depth
    4. Radix sort all keys in parallel on GPU (via gsplat's CUB DeviceRadixSort).
    5. Perform front-to-back alpha blending with transmittance accumulation.
       Dense static geometry rapidly saturates opacity; dynamic primitives
       behind static foreground are automatically occluded.

This approach is called "unified CUDA depth-sorting" and produces zero
compositing artifacts with full real-time performance.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from omnisplat4d.core.config import CompositeConfig
from omnisplat4d.core.types import CameraIntrinsics, GaussianCheckpoint

log = logging.getLogger(__name__)


def concat_gaussians(
    static: GaussianCheckpoint,
    dynamic: GaussianCheckpoint,
) -> GaussianCheckpoint:
    """
    Merge static background and dynamic subject Gaussian primitives.

    Both checkpoints must be in the same world coordinate system (guaranteed
    when both were trained from the same COLMAP workspace initialised by
    sfm/initializer.py).

    Args:
        static:  Trained static background GaussianCheckpoint.
        dynamic: Trained dynamic subject GaussianCheckpoint (canonical positions
                 already offset by the DeformationMLP output for frame t).

    Returns:
        Combined GaussianCheckpoint with N_static + N_dynamic primitives.
        sh_coeffs is None if either input uses MEGA DC-only mode.
    """
    def _cat(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
        if a is None or b is None:
            return None
        return np.concatenate([a, b], axis=0)

    return GaussianCheckpoint(
        positions=np.concatenate([static.positions, dynamic.positions], axis=0),
        rotations=np.concatenate([static.rotations, dynamic.rotations], axis=0),
        scales=np.concatenate([static.scales, dynamic.scales], axis=0),
        opacities=np.concatenate([static.opacities, dynamic.opacities], axis=0),
        dc_colors=np.concatenate([static.dc_colors, dynamic.dc_colors], axis=0),
        sh_coeffs=_cat(static.sh_coeffs, dynamic.sh_coeffs),
        source_path=None,
    )


def radix_sort_by_depth(
    gaussians: GaussianCheckpoint,
    camera_extrinsics: np.ndarray,
    tile_size: int = 16,
) -> torch.Tensor:
    """
    Compute 64-bit sort keys and return the depth-sorted index order.

    Sort key layout:
        Bits [63:32] = tile ID  (screen tile the Gaussian's projected centre falls in)
        Bits [31:0]  = float32 depth reinterpreted as uint32

    Args:
        gaussians:         Combined GaussianCheckpoint (static + dynamic).
        camera_extrinsics: (4, 4) world-to-camera transform matrix (float32).
        tile_size:         Tile dimension in pixels (must be 16 for gsplat).

    Returns:
        (N,) int64 tensor of sorted indices (depth-ascending, tile-major).

    Note:
        The actual radix sort is delegated to gsplat's internal CUDA kernel
        (CUB DeviceRadixSort). This function constructs the sort keys and
        calls the appropriate gsplat utility.
    """
    raise NotImplementedError(
        "radix_sort_by_depth: implement using gsplat's radix sort utilities. "
        "See gsplat source for the packed 64-bit key layout and tile ID computation."
    )


def render_frame(
    gaussians: GaussianCheckpoint,
    camera_intrinsics: CameraIntrinsics,
    camera_extrinsics: np.ndarray,
    cfg: CompositeConfig,
) -> np.ndarray:
    """
    Render one frame via unified depth-sorted alpha blending.

    Applies deformed dynamic Gaussian positions (caller is responsible for
    applying DeformationMLP offsets before passing the checkpoint here),
    concatenates with static background, sorts by depth, and blends.

    Args:
        gaussians:         Combined GaussianCheckpoint (from concat_gaussians).
        camera_intrinsics: Pinhole camera intrinsics for the render viewpoint.
        camera_extrinsics: (4, 4) world-to-camera extrinsics matrix.
        cfg:               CompositeConfig (tile_size, sort_key_bits).

    Returns:
        Rendered RGB image as uint8 numpy array (H, W, 3).
    """
    raise NotImplementedError(
        "render_frame: implement gsplat rasterisation with unified depth sort. "
        "See gsplat.rendering API for the rasterize_gaussians call signature."
    )
