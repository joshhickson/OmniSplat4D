"""
Phase 1 — Ingestion: Equirectangular-to-planar virtual camera projection.

Converts spherical 360 equirectangular frames into N overlapping rectilinear
planar crops, one per virtual camera arranged at equal yaw intervals around
the sphere. This bypasses polar distortion that causes standard SfM solvers
to fail on equirectangular imagery.

Key design:
    - Camera intrinsics are computed analytically from FOV — never estimated.
    - Rotation matrices are pre-computed once and reused for every frame.
    - cv2.remap() with BORDER_WRAP handles the east/west panorama seam.
    - Output images use the same resolution as the configured planar size.

Downstream consumers:
    - sfm/initializer.py reads CameraIntrinsics and rotation matrices to
      write cameras.txt and images.txt for programmatic COLMAP seeding.
    - segment/detector.py + masker.py operate on the planar crops, not on
      the raw equirectangular frames.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from omnisplat4d.core.config import IngestConfig
from omnisplat4d.core.types import CameraIntrinsics

log = logging.getLogger(__name__)


def build_virtual_cameras(cfg: IngestConfig) -> list[CameraIntrinsics]:
    """
    Compute intrinsic matrices for each virtual planar camera analytically.

    Cameras are arranged at equal yaw intervals: yaw_i = 360/N * i degrees.
    fx and fy are derived from the horizontal and vertical FOV respectively.
    Principal point is at the image centre.

    Args:
        cfg: IngestConfig with num_cameras, planar_fov_deg, planar_width, planar_height.

    Returns:
        List of CameraIntrinsics, one per virtual camera (len == cfg.num_cameras).
    """
    cameras: list[CameraIntrinsics] = []
    fov_rad = math.radians(cfg.planar_fov_deg)
    fx = cfg.planar_width / (2.0 * math.tan(fov_rad / 2.0))
    # Assume square pixels; vertical FOV derived from aspect ratio
    vfov_rad = 2.0 * math.atan(math.tan(fov_rad / 2.0) * cfg.planar_height / cfg.planar_width)
    fy = cfg.planar_height / (2.0 * math.tan(vfov_rad / 2.0))
    cx = cfg.planar_width / 2.0
    cy = cfg.planar_height / 2.0

    for i in range(cfg.num_cameras):
        cameras.append(
            CameraIntrinsics(
                camera_id=i,
                width=cfg.planar_width,
                height=cfg.planar_height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )
    return cameras


def build_rotation_matrices(cfg: IngestConfig) -> list[np.ndarray]:
    """
    Pre-compute 3×3 world-to-camera rotation matrices for each virtual camera.

    Camera i points at yaw_i = 360/N * i degrees around the Y (up) axis.
    Pitch and roll are zero — all cameras are on the equatorial plane.

    Args:
        cfg: IngestConfig with num_cameras.

    Returns:
        List of (3, 3) float64 numpy arrays, one per camera.
    """
    rotations: list[np.ndarray] = []
    for i in range(cfg.num_cameras):
        yaw_deg = 360.0 / cfg.num_cameras * i
        yaw_rad = math.radians(yaw_deg)
        # Rotation around Y axis (right-hand rule, camera looks along -Z in world)
        R = np.array([
            [ math.cos(yaw_rad), 0.0, math.sin(yaw_rad)],
            [              0.0, 1.0,              0.0],
            [-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
        ], dtype=np.float64)
        rotations.append(R)
    return rotations


def build_remap_grids(
    intrinsics: CameraIntrinsics,
    rotation_matrix: np.ndarray,
    equirect_width: int,
    equirect_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute cv2.remap() pixel coordinate maps for one virtual camera.

    For each output pixel (u, v):
        1. Un-project through the pinhole model to a 3D ray in camera space.
        2. Rotate into world space via R^T.
        3. Convert the 3D ray to spherical (phi, theta).
        4. Map to equirectangular pixel coordinates (x_src, y_src).

    Args:
        intrinsics:       Pinhole intrinsics for this camera.
        rotation_matrix:  3×3 world-to-camera rotation (R); inverse = R^T.
        equirect_width:   Width of the source equirectangular image.
        equirect_height:  Height of the source equirectangular image.

    Returns:
        (map_x, map_y): float32 arrays of shape (H_out, W_out) suitable for
        cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).
    """
    W, H = intrinsics.width, intrinsics.height
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    R_inv = rotation_matrix.T  # orthogonal matrix: inverse == transpose

    # Build pixel grid
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

    # Un-project to normalised camera rays
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = np.ones_like(x_cam)

    # Rotate rays into world space
    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    rays_world = rays @ R_inv.T  # (H, W, 3)

    # Normalise
    norms = np.linalg.norm(rays_world, axis=-1, keepdims=True)
    rays_world /= norms

    x_w, y_w, z_w = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]

    # Spherical coordinates
    lon = np.arctan2(x_w, z_w)               # [-pi, pi]
    lat = np.arcsin(np.clip(y_w, -1.0, 1.0)) # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    map_x = ((lon / math.pi + 1.0) / 2.0 * equirect_width).astype(np.float32)
    map_y = ((0.5 - lat / math.pi) * equirect_height).astype(np.float32)

    return map_x, map_y


def project_frame(
    equirect_bgr: np.ndarray,
    rotation_matrix: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Reproject one equirectangular frame into a planar crop for one virtual camera.

    Builds remap grids on-the-fly. For batch processing across many frames,
    pre-compute grids with build_remap_grids() and call cv2.remap() directly.

    Args:
        equirect_bgr:    Source equirectangular BGR image (H_src, W_src, 3).
        rotation_matrix: 3×3 world-to-camera rotation for this camera.
        intrinsics:      Pinhole intrinsics for this camera.

    Returns:
        Planar BGR crop of shape (intrinsics.height, intrinsics.width, 3).
    """
    H_src, W_src = equirect_bgr.shape[:2]
    map_x, map_y = build_remap_grids(intrinsics, rotation_matrix, W_src, H_src)
    return cv2.remap(equirect_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def project_all_frames(
    input_dir: Path,
    output_base_dir: Path,
    cameras: list[CameraIntrinsics],
    rotation_matrices: list[np.ndarray],
) -> list[list[Path]]:
    """
    Apply equirect-to-planar projection to all PNG frames in input_dir.

    Writes output to output_base_dir/cam_00/, cam_01/, … cam_NN/ directories,
    preserving the original filename within each camera subdirectory.

    Args:
        input_dir:        Directory containing equirectangular frame PNGs
                          (output of extractor.extract_frames).
        output_base_dir:  Root directory for per-camera output subdirectories.
        cameras:          List of CameraIntrinsics (from build_virtual_cameras).
        rotation_matrices: Corresponding rotation matrices (from build_rotation_matrices).

    Returns:
        List of lists: result[cam_idx][frame_idx] = Path to planar PNG.

    Side effects:
        Creates output_base_dir/cam_XX/ directories if they do not exist.
    """
    assert len(cameras) == len(rotation_matrices), "cameras and rotation_matrices must have equal length"

    frame_paths = sorted(input_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in {input_dir}")

    # Create output directories
    cam_dirs = [output_base_dir / f"cam_{i:02d}" for i in range(len(cameras))]
    for d in cam_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Pre-compute remap grids from the first frame's dimensions
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Could not read {frame_paths[0]}")
    H_src, W_src = sample.shape[:2]
    grids = [
        build_remap_grids(cam, rot, W_src, H_src)
        for cam, rot in zip(cameras, rotation_matrices)
    ]

    results: list[list[Path]] = [[] for _ in cameras]

    for fp in frame_paths:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            log.warning("Skipping unreadable frame: %s", fp)
            continue
        for cam_idx, (cam_dir, (map_x, map_y)) in enumerate(zip(cam_dirs, grids)):
            crop = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            out_path = cam_dir / fp.name
            cv2.imwrite(str(out_path), crop)
            results[cam_idx].append(out_path)

    log.info(
        "Projected %d frames × %d cameras → %s",
        len(frame_paths), len(cameras), output_base_dir,
    )
    return results
