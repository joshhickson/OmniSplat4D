"""
Tests for src/omnisplat4d/sfm/initializer.py

No GPU required — validates that cameras.txt and images.txt are written
in valid COLMAP text format and that the programmatic injection produces
geometrically correct poses.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnisplat4d.core.types import CameraIntrinsics, CameraPose
from omnisplat4d.sfm.initializer import (
    build_poses_from_rotations,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
)


@pytest.fixture
def sample_cameras() -> list[CameraIntrinsics]:
    return [
        CameraIntrinsics(camera_id=i, width=1024, height=1024, fx=512.0, fy=512.0, cx=512.0, cy=512.0)
        for i in range(4)
    ]


@pytest.fixture
def sample_rotations() -> list[np.ndarray]:
    return [np.eye(3) for _ in range(4)]


@pytest.fixture
def sample_frame_paths(tmp_path: Path) -> list[list[Path]]:
    paths = []
    for cam_idx in range(4):
        cam_frames = []
        for f in range(3):
            p = tmp_path / f"cam_{cam_idx:02d}" / f"frame_{f:06d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            cam_frames.append(p)
        paths.append(cam_frames)
    return paths


class TestWriteCamerasTxt:
    def test_file_is_created(self, sample_cameras: list[CameraIntrinsics], tmp_path: Path) -> None:
        out = tmp_path / "cameras.txt"
        write_cameras_txt(sample_cameras, out)
        assert out.exists()

    def test_correct_number_of_camera_lines(
        self, sample_cameras: list[CameraIntrinsics], tmp_path: Path
    ) -> None:
        out = tmp_path / "cameras.txt"
        write_cameras_txt(sample_cameras, out)
        data_lines = [
            l for l in out.read_text().splitlines() if l and not l.startswith("#")
        ]
        assert len(data_lines) == len(sample_cameras)

    def test_pinhole_model_in_each_line(
        self, sample_cameras: list[CameraIntrinsics], tmp_path: Path
    ) -> None:
        out = tmp_path / "cameras.txt"
        write_cameras_txt(sample_cameras, out)
        data_lines = [
            l for l in out.read_text().splitlines() if l and not l.startswith("#")
        ]
        for line in data_lines:
            parts = line.split()
            assert parts[1] == "PINHOLE", f"Expected PINHOLE model, got {parts[1]!r}"

    def test_camera_ids_match_intrinsics(
        self, sample_cameras: list[CameraIntrinsics], tmp_path: Path
    ) -> None:
        out = tmp_path / "cameras.txt"
        write_cameras_txt(sample_cameras, out)
        data_lines = [
            l for l in out.read_text().splitlines() if l and not l.startswith("#")
        ]
        for cam, line in zip(sample_cameras, data_lines):
            cam_id = int(line.split()[0])
            assert cam_id == cam.camera_id

    def test_focal_lengths_written_correctly(
        self, sample_cameras: list[CameraIntrinsics], tmp_path: Path
    ) -> None:
        out = tmp_path / "cameras.txt"
        write_cameras_txt(sample_cameras, out)
        data_lines = [
            l for l in out.read_text().splitlines() if l and not l.startswith("#")
        ]
        for cam, line in zip(sample_cameras, data_lines):
            parts = line.split()
            # PINHOLE: CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY
            fx = float(parts[4])
            fy = float(parts[5])
            assert fx == pytest.approx(cam.fx, rel=1e-4)
            assert fy == pytest.approx(cam.fy, rel=1e-4)


class TestWriteImagesTxt:
    def test_file_is_created(self, tmp_path: Path) -> None:
        poses = [
            CameraPose(
                image_id=1, camera_id=0,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                translation=np.zeros(3),
                image_name="cam_00/frame_000000.png",
            )
        ]
        out = tmp_path / "images.txt"
        write_images_txt(poses, out)
        assert out.exists()

    def test_each_image_has_two_lines(self, tmp_path: Path) -> None:
        """Each image entry = 1 data line + 1 empty keypoints line."""
        n_images = 5
        poses = [
            CameraPose(
                image_id=i + 1, camera_id=0,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                translation=np.zeros(3),
                image_name=f"cam_00/frame_{i:06d}.png",
            )
            for i in range(n_images)
        ]
        out = tmp_path / "images.txt"
        write_images_txt(poses, out)
        non_comment_lines = [
            l for l in out.read_text().splitlines() if not l.startswith("#")
        ]
        # Each image = 1 data line + 1 empty line = 2 per image
        assert len(non_comment_lines) == n_images * 2

    def test_translation_is_zero_for_virtual_cameras(self, tmp_path: Path) -> None:
        pose = CameraPose(
            image_id=1, camera_id=0,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            translation=np.zeros(3),
            image_name="cam_00/frame_000000.png",
        )
        out = tmp_path / "images.txt"
        write_images_txt([pose], out)
        data_line = next(
            l for l in out.read_text().splitlines()
            if l and not l.startswith("#")
        )
        parts = data_line.split()
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        assert tx == pytest.approx(0.0)
        assert ty == pytest.approx(0.0)
        assert tz == pytest.approx(0.0)


class TestBuildPosesFromRotations:
    def test_total_pose_count(
        self,
        sample_cameras: list[CameraIntrinsics],
        sample_rotations: list[np.ndarray],
        sample_frame_paths: list[list[Path]],
    ) -> None:
        poses = build_poses_from_rotations(sample_cameras, sample_rotations, sample_frame_paths)
        expected = len(sample_cameras) * len(sample_frame_paths[0])
        assert len(poses) == expected

    def test_image_ids_are_sequential_from_one(
        self,
        sample_cameras: list[CameraIntrinsics],
        sample_rotations: list[np.ndarray],
        sample_frame_paths: list[list[Path]],
    ) -> None:
        poses = build_poses_from_rotations(sample_cameras, sample_rotations, sample_frame_paths)
        ids = [p.image_id for p in poses]
        assert ids == list(range(1, len(poses) + 1))

    def test_quaternion_is_unit_length(
        self,
        sample_cameras: list[CameraIntrinsics],
        sample_rotations: list[np.ndarray],
        sample_frame_paths: list[list[Path]],
    ) -> None:
        poses = build_poses_from_rotations(sample_cameras, sample_rotations, sample_frame_paths)
        for pose in poses:
            norm = np.linalg.norm(pose.quaternion)
            assert norm == pytest.approx(1.0, abs=1e-6), f"Quaternion norm = {norm}"

    def test_translation_is_zero(
        self,
        sample_cameras: list[CameraIntrinsics],
        sample_rotations: list[np.ndarray],
        sample_frame_paths: list[list[Path]],
    ) -> None:
        poses = build_poses_from_rotations(sample_cameras, sample_rotations, sample_frame_paths)
        for pose in poses:
            assert np.allclose(pose.translation, 0.0)

    def test_identity_rotation_gives_canonical_quaternion(
        self,
        sample_cameras: list[CameraIntrinsics],
        sample_frame_paths: list[list[Path]],
    ) -> None:
        """Identity rotation matrix → quaternion [qw=1, qx=0, qy=0, qz=0]."""
        poses = build_poses_from_rotations(
            sample_cameras[:1],
            [np.eye(3)],
            sample_frame_paths[:1],
        )
        q = poses[0].quaternion  # [qw, qx, qy, qz]
        assert q[0] == pytest.approx(1.0, abs=1e-6)
        assert q[1] == pytest.approx(0.0, abs=1e-6)
        assert q[2] == pytest.approx(0.0, abs=1e-6)
        assert q[3] == pytest.approx(0.0, abs=1e-6)


class TestWritePoints3dTxt:
    def test_file_is_created(self, tmp_path: Path) -> None:
        out = tmp_path / "points3D.txt"
        write_points3d_txt(out)
        assert out.exists()

    def test_file_has_zero_points_comment(self, tmp_path: Path) -> None:
        out = tmp_path / "points3D.txt"
        write_points3d_txt(out)
        assert "Number of points: 0" in out.read_text()
