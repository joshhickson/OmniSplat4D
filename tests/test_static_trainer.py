from __future__ import annotations

from pathlib import Path

import numpy as np

from omnisplat4d.core.config import StaticTrainConfig
from omnisplat4d.train import static_trainer


def _write_points3d(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
                "1 0.0 0.0 0.0 255 0 0 0.1 1 2",
                "2 0.1 0.0 0.0 0 255 0 0.1 1 2",
                "3 0.0 0.1 0.0 0 0 255 0.1 1 2",
                "4 0.1 0.1 0.0 255 255 255 0.1 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_train_static_returns_fp16_for_spz_and_writes_background(tmp_path: Path, monkeypatch) -> None:
    colmap_dir = tmp_path / "colmap" / "sparse" / "0"
    _write_points3d(colmap_dir / "points3D.txt")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(static_trainer, "_resolve_gsplat_rasterizer", lambda: None)

    cfg = StaticTrainConfig(max_iterations=2, prune_interval=1, output_format="spz")
    ckpt = static_trainer.train_static(colmap_dir, None, cfg, output_dir)

    assert ckpt.positions.dtype == np.float16
    assert ckpt.opacities.dtype == np.float16
    assert (output_dir / "background_model.spz").exists()


def test_train_static_flushes_after_each_densification_step(tmp_path: Path, monkeypatch) -> None:
    colmap_dir = tmp_path / "colmap" / "sparse" / "0"
    _write_points3d(colmap_dir / "points3D.txt")
    output_dir = tmp_path / "out"

    calls = {"count": 0}

    def _flush_counter() -> None:
        calls["count"] += 1

    monkeypatch.setattr(static_trainer, "flush_cuda_cache", _flush_counter)
    monkeypatch.setattr(static_trainer, "_resolve_gsplat_rasterizer", lambda: None)

    cfg = StaticTrainConfig(max_iterations=4, prune_interval=1, output_format="ply")
    ckpt = static_trainer.train_static(colmap_dir, None, cfg, output_dir)

    assert calls["count"] == 4
    assert ckpt.positions.dtype == np.float32


def test_train_static_saves_checkpoint_every_5000_iterations(tmp_path: Path, monkeypatch) -> None:
    colmap_dir = tmp_path / "colmap" / "sparse" / "0"
    _write_points3d(colmap_dir / "points3D.txt")
    output_dir = tmp_path / "out"

    saved_steps: list[int] = []

    def _record_checkpoint(_ckpt, _out, step: int) -> None:
        saved_steps.append(step)

    monkeypatch.setattr(static_trainer, "_save_periodic_checkpoint", _record_checkpoint)
    monkeypatch.setattr(static_trainer, "_resolve_gsplat_rasterizer", lambda: None)

    cfg = StaticTrainConfig(max_iterations=5000, prune_interval=50, output_format="spz")
    static_trainer.train_static(colmap_dir, None, cfg, output_dir)

    assert saved_steps == [5000]
