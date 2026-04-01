"""
Phase 1/2 — Segment: Multi-view spherical identity continuity tracking.

When the dynamic subject moves from one virtual planar camera's frustum into
an adjacent camera's frustum, this module performs a geometric handoff to
maintain a single continuous object identity across the full 360 sphere.

Algorithm (Geometric Handoff):
    1. Project the 2D bounding box from Camera A into a 3D ray on the unit sphere
       using the inverse pinhole model and Camera A's rotation matrix.
    2. Rotate that 3D ray into Camera B's camera space using Camera B's rotation.
    3. Project the 3D ray back to a 2D bounding box in Camera B's image.
    4. If the reprojected box lies within Camera B's image bounds, pass it as
       a SAM 2.1 prompt for the handoff frame.

This geometric approach avoids re-running YOLO detection at the boundary,
ensuring continuity when the subject is partially visible across two cameras.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from omnisplat4d.core.types import CameraIntrinsics

log = logging.getLogger(__name__)


def project_bbox_to_sphere(
    bbox: np.ndarray,
    intrinsics: CameraIntrinsics,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Project a 2D bounding box from a planar camera image into 3D unit sphere vectors.

    The bounding box corners are un-projected through the pinhole model into
    normalised 3D rays in camera space, then rotated into world space.

    Args:
        bbox:             Bounding box [x1, y1, x2, y2] in image pixel coordinates.
        intrinsics:       Pinhole intrinsics for the source camera.
        rotation_matrix:  3×3 world-to-camera rotation R for the source camera.
                          World rays = R^T @ camera_rays.

    Returns:
        Array of shape (4, 3) — four unit sphere vectors corresponding to the
        four corners [top-left, top-right, bottom-right, bottom-left].
    """
    x1, y1, x2, y2 = bbox
    corners_img = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

    R_inv = rotation_matrix.T
    sphere_vecs = np.zeros((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_img):
        ray_cam = np.array(
            [(u - intrinsics.cx) / intrinsics.fx,
             (v - intrinsics.cy) / intrinsics.fy,
             1.0],
            dtype=np.float64,
        )
        ray_world = R_inv @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        sphere_vecs[i] = ray_world

    return sphere_vecs


def reproject_sphere_to_camera(
    sphere_vecs: np.ndarray,
    target_intrinsics: CameraIntrinsics,
    target_rotation: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Project 3D unit sphere vectors into a target camera's image space.

    Args:
        sphere_vecs:      (4, 3) unit sphere vectors (world space).
        target_intrinsics: Pinhole intrinsics for the target camera.
        target_rotation:  3×3 world-to-camera rotation R for the target camera.

    Returns:
        Bounding box [x1, y1, x2, y2] in target camera pixel coordinates,
        or None if any corner projects behind the camera (z <= 0) or entirely
        outside the image bounds.
    """
    projected_pts = []
    for vec_world in sphere_vecs:
        ray_cam = target_rotation @ vec_world
        if ray_cam[2] <= 0.0:
            return None  # behind camera
        u = target_intrinsics.fx * ray_cam[0] / ray_cam[2] + target_intrinsics.cx
        v = target_intrinsics.fy * ray_cam[1] / ray_cam[2] + target_intrinsics.cy
        projected_pts.append((u, v))

    us = [p[0] for p in projected_pts]
    vs = [p[1] for p in projected_pts]
    x1, x2 = min(us), max(us)
    y1, y2 = min(vs), max(vs)

    # Reject if entirely outside image bounds
    W, H = target_intrinsics.width, target_intrinsics.height
    if x2 < 0 or x1 > W or y2 < 0 or y1 > H:
        return None

    # Clamp to image bounds
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(W), x2)
    y2 = min(float(H), y2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def handoff_tracking(
    bbox_cam_a: np.ndarray,
    cam_a_intrinsics: CameraIntrinsics,
    cam_a_rotation: np.ndarray,
    cam_b_intrinsics: CameraIntrinsics,
    cam_b_rotation: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Full geometric handoff: transfer a bounding box from Camera A to Camera B.

    Combines project_bbox_to_sphere() and reproject_sphere_to_camera() into
    a single call for the common cross-camera boundary case.

    Args:
        bbox_cam_a:        Bounding box in Camera A's image space [x1,y1,x2,y2].
        cam_a_intrinsics:  Intrinsics for Camera A.
        cam_a_rotation:    Rotation matrix for Camera A.
        cam_b_intrinsics:  Intrinsics for Camera B.
        cam_b_rotation:    Rotation matrix for Camera B.

    Returns:
        Bounding box [x1, y1, x2, y2] in Camera B's image space,
        or None if the subject is not visible in Camera B.
    """
    sphere_vecs = project_bbox_to_sphere(bbox_cam_a, cam_a_intrinsics, cam_a_rotation)
    return reproject_sphere_to_camera(sphere_vecs, cam_b_intrinsics, cam_b_rotation)


def warp_bbox_optical_flow(
    bbox: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """
    Warp a bounding box using a pre-computed dense optical flow field.

    Used to propagate the YOLO bounding box prompt across frames without
    re-running inference every frame. Samples the flow at the four corners
    and the box centre, then recomputes the enclosing box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        flow: Dense optical flow array of shape (H, W, 2) from Farneback.
              flow[y, x] = (dx, dy) pixel displacement.

    Returns:
        Warped bounding box [x1, y1, x2, y2] as float32.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    H, W = flow.shape[:2]

    sample_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [cx, cy]], dtype=np.int32)
    sample_pts[:, 0] = np.clip(sample_pts[:, 0], 0, W - 1)
    sample_pts[:, 1] = np.clip(sample_pts[:, 1], 0, H - 1)

    displacements = flow[sample_pts[:, 1], sample_pts[:, 0]]  # (5, 2)
    warped = sample_pts.astype(np.float32) + displacements

    new_x1 = float(warped[:, 0].min())
    new_y1 = float(warped[:, 1].min())
    new_x2 = float(warped[:, 0].max())
    new_y2 = float(warped[:, 1].max())

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
