"""
Tests for src/omnisplat4d/ingest/projector.py

No GPU required — all tests exercise pure math (trigonometry, matrix ops).
Validates the equirectangular-to-planar reprojection pipeline before any
GPU hardware is involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omnisplat4d.core.config import IngestConfig
from omnisplat4d.ingest.projector import (
    build_rotation_matrices,
    build_remap_grids,
    build_virtual_cameras,
    project_frame,
)


@pytest.fixture
def cfg() -> IngestConfig:
    return IngestConfig(
        num_cameras=8,
        planar_fov_deg=90.0,
        planar_width=256,
        planar_height=256,
    )


class TestBuildVirtualCameras:
    def test_returns_correct_count(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        assert len(cameras) == cfg.num_cameras

    def test_camera_ids_are_sequential(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        for i, cam in enumerate(cameras):
            assert cam.camera_id == i

    def test_intrinsics_dimensions_match_config(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        for cam in cameras:
            assert cam.width == cfg.planar_width
            assert cam.height == cfg.planar_height

    def test_principal_point_at_image_centre(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        for cam in cameras:
            assert cam.cx == pytest.approx(cfg.planar_width / 2.0)
            assert cam.cy == pytest.approx(cfg.planar_height / 2.0)

    def test_focal_length_consistent_with_fov(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        expected_fx = cfg.planar_width / (2.0 * math.tan(math.radians(cfg.planar_fov_deg) / 2.0))
        for cam in cameras:
            assert cam.fx == pytest.approx(expected_fx, rel=1e-5)

    def test_all_cameras_identical_intrinsics(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        for cam in cameras[1:]:
            assert cam.fx == pytest.approx(cameras[0].fx)
            assert cam.fy == pytest.approx(cameras[0].fy)


class TestBuildRotationMatrices:
    def test_returns_correct_count(self, cfg: IngestConfig) -> None:
        rotations = build_rotation_matrices(cfg)
        assert len(rotations) == cfg.num_cameras

    def test_all_rotation_matrices_are_orthogonal(self, cfg: IngestConfig) -> None:
        rotations = build_rotation_matrices(cfg)
        for R in rotations:
            RRT = R @ R.T
            assert np.allclose(RRT, np.eye(3), atol=1e-10), f"R @ R.T is not identity:\n{RRT}"

    def test_all_rotation_matrices_have_unit_determinant(self, cfg: IngestConfig) -> None:
        rotations = build_rotation_matrices(cfg)
        for R in rotations:
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-10, f"det(R) = {det}, expected 1.0"

    def test_first_camera_is_identity_direction(self, cfg: IngestConfig) -> None:
        """Camera 0 at yaw=0 should look along +Z (forward direction)."""
        rotations = build_rotation_matrices(cfg)
        R0 = rotations[0]
        assert np.allclose(R0, np.eye(3), atol=1e-10), f"Camera 0 rotation:\n{R0}"

    def test_cameras_are_evenly_spaced_in_yaw(self, cfg: IngestConfig) -> None:
        """Adjacent cameras should differ by 360/N degrees in yaw."""
        rotations = build_rotation_matrices(cfg)
        expected_yaw_step = 2 * math.pi / cfg.num_cameras
        for i in range(1, len(rotations)):
            # Relative rotation between consecutive cameras
            R_rel = rotations[i] @ rotations[i - 1].T
            # Trace encodes 1 + 2*cos(theta) for rotation matrices
            cos_theta = (np.trace(R_rel) - 1.0) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = math.acos(cos_theta)
            assert angle == pytest.approx(expected_yaw_step, abs=1e-8), (
                f"Camera {i} yaw step {math.degrees(angle):.2f}° != "
                f"{math.degrees(expected_yaw_step):.2f}°"
            )


class TestBuildRemapGrids:
    def test_output_shapes_match_intrinsics(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        rotations = build_rotation_matrices(cfg)
        equirect_w, equirect_h = 1024, 512
        map_x, map_y = build_remap_grids(cameras[0], rotations[0], equirect_w, equirect_h)
        assert map_x.shape == (cfg.planar_height, cfg.planar_width)
        assert map_y.shape == (cfg.planar_height, cfg.planar_width)

    def test_output_dtype_is_float32(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        rotations = build_rotation_matrices(cfg)
        map_x, map_y = build_remap_grids(cameras[0], rotations[0], 1024, 512)
        assert map_x.dtype == np.float32
        assert map_y.dtype == np.float32

    def test_source_coordinates_within_equirect_bounds(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        rotations = build_rotation_matrices(cfg)
        equirect_w, equirect_h = 1024, 512
        map_x, map_y = build_remap_grids(cameras[0], rotations[0], equirect_w, equirect_h)
        # map_x can wrap (BORDER_WRAP), map_y should be within [0, H)
        assert map_y.min() >= 0
        assert map_y.max() < equirect_h


class TestProjectFrame:
    def test_output_shape_matches_intrinsics(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        rotations = build_rotation_matrices(cfg)
        equirect = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        result = project_frame(equirect, rotations[0], cameras[0])
        assert result.shape == (cfg.planar_height, cfg.planar_width, 3)

    def test_output_dtype_is_uint8(self, cfg: IngestConfig) -> None:
        cameras = build_virtual_cameras(cfg)
        rotations = build_rotation_matrices(cfg)
        equirect = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        result = project_frame(equirect, rotations[0], cameras[0])
        assert result.dtype == np.uint8
