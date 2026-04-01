"""
Phase 1 — SfM: Programmatic COLMAP workspace initialisation.

Instead of running `colmap feature_extractor` (which re-estimates camera models
from image content), this module directly writes the cameras.txt and images.txt
files using analytically-derived intrinsics and rotation-derived quaternions.

Why this matters:
    - Eliminates COLMAP's expensive SIFT feature extraction pass entirely.
    - Locks intra-frame camera relationships: all 8 virtual cameras from a
      single equirectangular frame share the same optical centre (translation = 0).
    - Ensures the intrinsic K-matrices are exact (from projector.build_virtual_cameras)
      rather than estimated from sparse feature correspondences.
    - Makes the reconstruction deterministic and reproducible.

COLMAP text format references:
    cameras.txt:   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    images.txt:    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    points3D.txt:  Empty at initialisation — COLMAP's mapper fills this in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from omnisplat4d.core.config import ColmapConfig
from omnisplat4d.core.types import CameraIntrinsics, CameraPose

log = logging.getLogger(__name__)


def write_cameras_txt(
    cameras: list[CameraIntrinsics],
    output_path: Path,
) -> None:
    """
    Write cameras.txt in COLMAP PINHOLE format.

    Each virtual planar camera is written as a separate PINHOLE camera model
    entry. PINHOLE model parameters: fx fy cx cy (4 values).

    Args:
        cameras:     List of CameraIntrinsics (from projector.build_virtual_cameras).
        output_path: Destination path for cameras.txt.

    Side effects:
        Creates parent directories if they do not exist.
        Overwrites output_path if it exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for cam in cameras:
        params = f"{cam.fx:.6f} {cam.fy:.6f} {cam.cx:.6f} {cam.cy:.6f}"
        lines.append(f"{cam.camera_id} PINHOLE {cam.width} {cam.height} {params}")
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %d cameras to %s", len(cameras), output_path)


def write_images_txt(
    poses: list[CameraPose],
    output_path: Path,
) -> None:
    """
    Write images.txt in COLMAP format.

    Each pose is written as two lines:
        Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        Line 2: (empty — no 2D keypoints at this stage)

    Args:
        poses:       List of CameraPose (from build_poses_from_rotations).
        output_path: Destination path for images.txt.

    Side effects:
        Creates parent directories if they do not exist.
        Overwrites output_path if it exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(poses)}",
    ]
    for pose in poses:
        qw, qx, qy, qz = pose.quaternion
        tx, ty, tz = pose.translation
        lines.append(
            f"{pose.image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.6f} {ty:.6f} {tz:.6f} {pose.camera_id} {pose.image_name}"
        )
        lines.append("")  # empty 2D keypoints line
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %d image poses to %s", len(poses), output_path)


def write_points3d_txt(output_path: Path) -> None:
    """
    Write an empty points3D.txt file.

    COLMAP's mapper requires this file to exist even if it is empty.
    The mapper will populate it with 3D points from feature matching.

    Args:
        output_path: Destination path for points3D.txt.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "# Number of points: 0\n"
    )


def build_poses_from_rotations(
    cameras: list[CameraIntrinsics],
    rotation_matrices: list[np.ndarray],
    frame_paths_per_camera: list[list[Path]],
) -> list[CameraPose]:
    """
    Build CameraPose objects from rotation matrices and planar frame paths.

    For virtual cameras sharing a single optical centre (drone route and stick
    route both share the 360 camera's physical centre), translation is always
    [0, 0, 0]. Quaternions are converted from rotation matrices using scipy.

    Image names follow the COLMAP convention: relative path from the images root,
    formatted as cam_XX/frame_XXXXXX.png.

    Args:
        cameras:               List of CameraIntrinsics.
        rotation_matrices:     Corresponding 3×3 rotation matrices (world-to-camera).
        frame_paths_per_camera: frame_paths_per_camera[cam_idx][frame_idx] = absolute Path.
                                All cameras must have the same number of frames.

    Returns:
        Flat list of CameraPose, one per (camera, frame) pair.
        image_id is a sequential 1-based integer (COLMAP convention).
    """
    assert len(cameras) == len(rotation_matrices) == len(frame_paths_per_camera)
    num_frames = len(frame_paths_per_camera[0])
    poses: list[CameraPose] = []
    image_id = 1

    for cam_idx, (cam, R, frame_paths) in enumerate(
        zip(cameras, rotation_matrices, frame_paths_per_camera)
    ):
        rot = Rotation.from_matrix(R)
        # scipy returns [x, y, z, w]; COLMAP wants [qw, qx, qy, qz]
        xyzw = rot.as_quat()
        quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)

        for frame_path in frame_paths:
            image_name = f"cam_{cam_idx:02d}/{frame_path.name}"
            poses.append(
                CameraPose(
                    image_id=image_id,
                    camera_id=cam.camera_id,
                    quaternion=quat.copy(),
                    translation=translation.copy(),
                    image_name=image_name,
                )
            )
            image_id += 1

    return poses


def initialize_colmap_workspace(
    output_dir: Path,
    cameras: list[CameraIntrinsics],
    rotation_matrices: list[np.ndarray],
    frame_paths_per_camera: list[list[Path]],
    cfg: ColmapConfig,
) -> Path:
    """
    Create a fully seeded COLMAP sparse/0/ workspace without running feature_extractor.

    Writes cameras.txt, images.txt, and an empty points3D.txt into
    output_dir/sparse/0/. The COLMAP runner can then invoke the matcher
    and mapper directly against this pre-seeded workspace.

    Args:
        output_dir:            Root COLMAP workspace directory (e.g. workspace/colmap_data/).
        cameras:               Virtual camera intrinsics list.
        rotation_matrices:     Corresponding rotation matrices.
        frame_paths_per_camera: Per-camera lists of planar PNG paths.
        cfg:                   ColmapConfig (used for logging / future options).

    Returns:
        Path to the sparse/0/ directory that was created.

    Raises:
        AssertionError: If cfg.skip_feature_extraction is False
                        (this function must only be called when bypassing COLMAP extraction).
    """
    assert cfg.skip_feature_extraction, (
        "initialize_colmap_workspace must only be called when skip_feature_extraction=True. "
        "Set ColmapConfig.skip_feature_extraction=True in your config."
    )

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    poses = build_poses_from_rotations(cameras, rotation_matrices, frame_paths_per_camera)
    write_cameras_txt(cameras, sparse_dir / "cameras.txt")
    write_images_txt(poses, sparse_dir / "images.txt")
    write_points3d_txt(sparse_dir / "points3D.txt")

    log.info(
        "Initialised COLMAP workspace: %d cameras, %d images → %s",
        len(cameras), len(poses), sparse_dir,
    )
    return sparse_dir
