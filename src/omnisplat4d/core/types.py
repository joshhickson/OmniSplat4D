"""
Shared data structures used as typed contracts between pipeline DAG nodes.

All dataclasses here are pure data holders — no business logic.
Import from this module when passing state between pipeline phases to avoid
circular imports between sub-packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic matrix for one virtual planar camera.

    All values in pixels. Derived analytically from FOV and resolution —
    never estimated by COLMAP for virtual cameras. Used in projector.py to
    construct the remap grids and in initializer.py to write cameras.txt.
    """

    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraPose:
    """
    Extrinsic pose (world-to-camera) for a single image observation.

    quaternion:  [qw, qx, qy, qz] in COLMAP convention.
    translation: [tx, ty, tz]. Always [0, 0, 0] for virtual cameras that
                 share a single physical optical center (drone / stick route).
    """

    image_id: int
    camera_id: int
    quaternion: np.ndarray   # shape (4,) — [qw, qx, qy, qz]
    translation: np.ndarray  # shape (3,)
    image_name: str


@dataclass
class FrameBatch:
    """
    A batch of extracted planar frames for one temporal sliding window.

    Attributes:
        frame_indices: Original source frame numbers from the decoded video.
        camera_dirs:   One directory per virtual camera, each containing the
                       planar PNGs for this batch. len == num_cameras.
        mask_dirs:     Corresponding mask directories. None for Drone Route
                       (no operator masking needed).
    """

    frame_indices: list[int]
    camera_dirs: list[Path]
    mask_dirs: Optional[list[Path]] = None


@dataclass
class GaussianCheckpoint:
    """
    Serialized state of a trained set of Gaussian primitives.

    Serves as the output contract from static_trainer and dynamic_trainer,
    and as the inter-window handoff object in the SWinGS sliding-window loop.

    All arrays are float16 when checkpoint_fp16=True (default for 12GB hardware).

    Attributes:
        positions:   (N, 3) — Gaussian centre positions in world space.
        rotations:   (N, 4) — Unit quaternions [qw, qx, qy, qz].
        scales:      (N, 3) — Log-scale values along each axis.
        opacities:   (N, 1) — Pre-sigmoid opacity values.
        dc_colors:   (N, 3) — DC (base RGB) color coefficients.
        sh_coeffs:   (N, K) — Higher-degree SH coefficients. None when
                               dc_color_only=True (MEGA mode).
        source_path: Path to the .spz file this was loaded from, if any.
    """

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    dc_colors: np.ndarray
    sh_coeffs: Optional[np.ndarray] = None
    source_path: Optional[Path] = None
