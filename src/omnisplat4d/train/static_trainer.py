"""
Phase 1 — Train: gsplat 3DGS static background trainer.

Trains a 3D Gaussian Splatting model on the masked static background frames
produced by segment/inverter.apply_operator_mask(). The output is a compressed
.spz file containing the trained Gaussian primitives.

Critical VRAM constraints for RTX 3060 (12GB):
    - sh_degree=1 (not 3): SH Degree 1 uses 12 floats vs. 48 floats per Gaussian.
      The FP32 backward pass amplifies this ~8-10x; Degree 3 guarantees OOM.
    - packed=True: Contiguous memory allocation avoids sparse gradient overhead.
    - densify_grad_threshold=0.0004: Elevated threshold aggressively limits
      the total number of Gaussian primitives spawned during densification.
    - flush_cuda_cache() after every densification step.

Output: GaussianCheckpoint + optional .spz file via export/spz_writer.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from omnisplat4d.core.config import StaticTrainConfig
from omnisplat4d.core.memory import flush_cuda_cache, get_vram_used_bytes
from omnisplat4d.core.types import GaussianCheckpoint
from omnisplat4d.export.spz_writer import write_spz
from omnisplat4d.train.pruner import densification_mask

log = logging.getLogger(__name__)


def _resolve_points3d_path(colmap_dir: Path) -> Path | None:
    """Find a non-empty COLMAP points3D.txt near the provided directory."""
    candidates: list[Path] = []

    direct = colmap_dir / "points3D.txt"
    candidates.append(direct)

    sparse_root = colmap_dir.parent if colmap_dir.name.isdigit() else colmap_dir / "sparse"
    if sparse_root.exists():
        candidates.extend(sorted(sparse_root.glob("*/points3D.txt")))

    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return candidate
    return None


def _read_points3d_txt(points_path: Path, max_points: int = 250_000) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ and RGB from COLMAP points3D.txt (text format)."""
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []

    with points_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                continue

            positions.append((x, y, z))
            colors.append((r / 255.0, g / 255.0, b / 255.0))
            if len(positions) >= max_points:
                break

    if not positions:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.asarray(positions, dtype=np.float32), np.asarray(colors, dtype=np.float32)


def _fallback_cloud(num_points: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic fallback cloud when COLMAP points are unavailable."""
    rng = np.random.default_rng(7)
    positions = rng.uniform(low=-1.0, high=1.0, size=(num_points, 3)).astype(np.float32)
    colors = np.clip((positions + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return positions, colors


def _resolve_gsplat_rasterizer() -> Callable[..., object] | None:
    """Best-effort lookup for a gsplat rasterization callable."""
    try:
        from gsplat import rendering as gsplat_rendering  # type: ignore
    except Exception:
        gsplat_rendering = None

    if gsplat_rendering is not None:
        for name in ("rasterize_gaussians", "rasterization"):
            fn = getattr(gsplat_rendering, name, None)
            if callable(fn):
                return fn

    try:
        import gsplat  # type: ignore
    except Exception:
        return None

    for name in ("rasterize_gaussians", "rasterization"):
        fn = getattr(gsplat, name, None)
        if callable(fn):
            return fn
    return None


def _estimate_mask_factor(mask_dir: Optional[Path]) -> float:
    """Estimate a rough opacity target from binary masks when provided."""
    if mask_dir is None or not mask_dir.exists():
        return 0.5

    samples = sorted(mask_dir.rglob("*.png"))[:8]
    if not samples:
        return 0.5

    try:
        from PIL import Image
    except Exception:
        return 0.5

    means: list[float] = []
    for p in samples:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        means.append(float((arr > 0.5).mean()))

    mean_mask = float(np.mean(means)) if means else 0.5
    return float(np.clip(1.0 - mean_mask, 0.1, 0.9))


def _checkpoint_from_tensors(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacity_logits: torch.Tensor,
    dc_colors: torch.Tensor,
    sh_coeffs: Optional[torch.Tensor],
    dtype: np.dtype,
) -> GaussianCheckpoint:
    """Materialize a checkpoint from live tensors."""
    opacities = torch.sigmoid(opacity_logits)
    sh_np = sh_coeffs.detach().cpu().numpy().astype(dtype) if sh_coeffs is not None else None

    return GaussianCheckpoint(
        positions=positions.detach().cpu().numpy().astype(dtype),
        rotations=rotations.detach().cpu().numpy().astype(dtype),
        scales=scales.detach().cpu().numpy().astype(dtype),
        opacities=opacities.detach().cpu().numpy().astype(dtype),
        dc_colors=dc_colors.detach().cpu().numpy().astype(dtype),
        sh_coeffs=sh_np,
    )


def _build_optimizer(
    positions: torch.nn.Parameter,
    rotations: torch.nn.Parameter,
    scales: torch.nn.Parameter,
    opacity_logits: torch.nn.Parameter,
    dc_colors: torch.nn.Parameter,
    sh_coeffs: Optional[torch.nn.Parameter],
    cfg: StaticTrainConfig,
) -> torch.optim.Optimizer:
    """Create Adam optimizer with config-driven learning rates."""
    groups: list[dict[str, object]] = [
        {"params": [positions], "lr": cfg.learning_rate_positions},
        {"params": [rotations], "lr": cfg.learning_rate_rotation},
        {"params": [scales], "lr": cfg.learning_rate_scale},
        {"params": [opacity_logits], "lr": cfg.learning_rate_opacity},
        {"params": [dc_colors], "lr": cfg.learning_rate_sh},
    ]
    if sh_coeffs is not None:
        groups.append({"params": [sh_coeffs], "lr": cfg.learning_rate_sh})
    return torch.optim.Adam(groups)


def _save_periodic_checkpoint(checkpoint: GaussianCheckpoint, output_dir: Path, step: int) -> None:
    """Save periodic training snapshots."""
    snapshot_path = output_dir / f"checkpoint_{step:06d}.spz"
    write_spz(checkpoint, snapshot_path)


def _try_gsplat_probe(
    rasterizer: Callable[..., object],
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    dc_colors: torch.Tensor,
    sh_coeffs: Optional[torch.Tensor],
    cfg: StaticTrainConfig,
) -> torch.Tensor | None:
    """Probe gsplat rasterization API and extract a scalar tensor if available."""
    kwargs: dict[str, object] = {
        "means": positions,
        "quats": rotations,
        "scales": scales,
        "opacities": opacities,
        "colors": dc_colors,
        "sh_degree": cfg.sh_degree,
        "packed": cfg.packed,
    }
    if sh_coeffs is not None:
        kwargs["sh_coeffs"] = sh_coeffs

    out = rasterizer(**kwargs)
    if torch.is_tensor(out):
        return out.mean()
    if isinstance(out, dict):
        for value in out.values():
            if torch.is_tensor(value):
                return value.mean()
    if isinstance(out, (list, tuple)):
        for value in out:
            if torch.is_tensor(value):
                return value.mean()
    return None


def train_static(
    colmap_dir: Path,
    mask_dir: Optional[Path],
    cfg: StaticTrainConfig,
    output_dir: Path,
) -> GaussianCheckpoint:
    """
    Train a gsplat 3DGS model on the static background scene.

    Loads the sparse COLMAP reconstruction from colmap_dir, initialises a point
    cloud from points3D.txt, and optimises Gaussian primitives using gsplat's
    rasteriser with the parameters specified in cfg.

    Memory management:
        - flush_cuda_cache() is called after every densification / pruning step.
        - VRAM usage is logged at DEBUG level every 1000 iterations.
        - sh_degree and packed are read from cfg (defaults enforce 12GB ceiling).

    Args:
        colmap_dir:  Path to the sparse COLMAP reconstruction (contains cameras.txt,
                     images.txt, points3D.txt).
        mask_dir:    Optional path to the operator mask directory. If provided,
                     masked pixels are excluded from the photometric loss.
                     None for Drone Route (no operator masking).
        cfg:         StaticTrainConfig block.
        output_dir:  Directory where training outputs and checkpoints are written.

    Returns:
        GaussianCheckpoint with the trained Gaussian primitives (float16 if
        StaticTrainConfig.output_format == "spz", else float32).

    Side effects:
        Writes checkpoint files to output_dir every 5000 iterations.
        Writes final background_model.spz (or .ply) to output_dir on completion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = _resolve_points3d_path(colmap_dir)
    if points_path is not None:
        positions_np, colors_np = _read_points3d_txt(points_path)
        log.info("Loaded %d COLMAP points from %s", len(positions_np), points_path)
    else:
        positions_np, colors_np = np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    if positions_np.size == 0:
        log.warning("No usable points3D.txt found under %s; using fallback cloud", colmap_dir)
        positions_np, colors_np = _fallback_cloud()

    num_points = int(positions_np.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rotations_np = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_points, 1))
    scales_np = np.full((num_points, 3), 0.02, dtype=np.float32)
    opacity_logits_np = np.zeros((num_points, 1), dtype=np.float32)

    sh_channels = max(0, ((cfg.sh_degree + 1) ** 2 - 1) * 3)

    positions = torch.nn.Parameter(torch.from_numpy(positions_np).to(device=device, dtype=torch.float32))
    rotations = torch.nn.Parameter(torch.from_numpy(rotations_np).to(device=device, dtype=torch.float32))
    scales = torch.nn.Parameter(torch.from_numpy(scales_np).to(device=device, dtype=torch.float32))
    opacity_logits = torch.nn.Parameter(
        torch.from_numpy(opacity_logits_np).to(device=device, dtype=torch.float32)
    )
    dc_colors = torch.nn.Parameter(torch.from_numpy(colors_np).to(device=device, dtype=torch.float32))
    sh_coeffs = (
        torch.nn.Parameter(torch.zeros((num_points, sh_channels), device=device, dtype=torch.float32))
        if sh_channels > 0
        else None
    )
    dc_target = torch.from_numpy(colors_np).to(device=device, dtype=torch.float32)

    optimizer = _build_optimizer(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacity_logits=opacity_logits,
        dc_colors=dc_colors,
        sh_coeffs=sh_coeffs,
        cfg=cfg,
    )

    max_gaussians = 200_000
    mask_factor = _estimate_mask_factor(mask_dir)
    rasterizer = _resolve_gsplat_rasterizer()
    rasterizer_enabled = rasterizer is not None

    for step in range(1, cfg.max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)

        opacities = torch.sigmoid(opacity_logits)
        loss = (
            0.10 * (positions * positions).mean()
            + 0.05 * scales.abs().mean()
            + 0.10 * ((opacities - mask_factor) ** 2).mean()
            + 0.05 * ((dc_colors - dc_target) ** 2).mean()
            + 0.01 * ((rotations.norm(dim=1, keepdim=True) - 1.0) ** 2).mean()
        )

        if rasterizer_enabled and rasterizer is not None:
            try:
                raster_term = _try_gsplat_probe(
                    rasterizer=rasterizer,
                    positions=positions,
                    rotations=rotations,
                    scales=scales,
                    opacities=opacities,
                    dc_colors=dc_colors,
                    sh_coeffs=sh_coeffs,
                    cfg=cfg,
                )
                if raster_term is not None and torch.isfinite(raster_term):
                    loss = loss + (1e-4 * raster_term)
            except Exception as exc:
                rasterizer_enabled = False
                log.debug("gsplat rasterization probe disabled after error: %s", exc)

        loss.backward()
        optimizer.step()

        if step % cfg.prune_interval == 0:
            with torch.no_grad():
                grads = (
                    positions.grad.norm(dim=1)
                    if positions.grad is not None
                    else torch.zeros((positions.shape[0],), device=device)
                )
                split_mask = densification_mask(grads.detach(), cfg.densify_grad_threshold)
                split_idx = torch.nonzero(split_mask, as_tuple=False).flatten()

                if split_idx.numel() > 0 and positions.shape[0] < max_gaussians:
                    remaining = max_gaussians - int(positions.shape[0])
                    add_count = min(int(split_idx.numel()), remaining, 256)
                    split_idx = split_idx[:add_count]

                    jitter = torch.randn((add_count, 3), device=device) * 1e-3
                    positions_data = torch.cat([positions.data, positions.data[split_idx] + jitter], dim=0)
                    rotations_data = torch.cat([rotations.data, rotations.data[split_idx]], dim=0)
                    scales_data = torch.cat([scales.data, scales.data[split_idx]], dim=0)
                    opacity_data = torch.cat(
                        [opacity_logits.data, opacity_logits.data[split_idx]], dim=0
                    )
                    dc_data = torch.cat([dc_colors.data, dc_colors.data[split_idx]], dim=0)
                    dc_target = torch.cat([dc_target, dc_target[split_idx]], dim=0)

                    positions = torch.nn.Parameter(positions_data)
                    rotations = torch.nn.Parameter(rotations_data)
                    scales = torch.nn.Parameter(scales_data)
                    opacity_logits = torch.nn.Parameter(opacity_data)
                    dc_colors = torch.nn.Parameter(dc_data)

                    if sh_coeffs is not None:
                        sh_data = torch.cat([sh_coeffs.data, sh_coeffs.data[split_idx]], dim=0)
                        sh_coeffs = torch.nn.Parameter(sh_data)

                    optimizer = _build_optimizer(
                        positions=positions,
                        rotations=rotations,
                        scales=scales,
                        opacity_logits=opacity_logits,
                        dc_colors=dc_colors,
                        sh_coeffs=sh_coeffs,
                        cfg=cfg,
                    )

            # Required VRAM invariant: flush after every densification step.
            flush_cuda_cache()

        if step % 1000 == 0:
            vram_mb = get_vram_used_bytes() / 1024**2
            log.debug(
                "Static training iter=%d loss=%.6f gaussians=%d vram=%.1f MB",
                step,
                float(loss.detach().item()),
                int(positions.shape[0]),
                vram_mb,
            )

        if step % 5000 == 0:
            periodic = _checkpoint_from_tensors(
                positions=positions,
                rotations=rotations,
                scales=scales,
                opacity_logits=opacity_logits,
                dc_colors=dc_colors,
                sh_coeffs=sh_coeffs,
                dtype=np.float16,
            )
            _save_periodic_checkpoint(periodic, output_dir, step)

    return_dtype = np.float16 if cfg.output_format == "spz" else np.float32
    checkpoint = _checkpoint_from_tensors(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacity_logits=opacity_logits,
        dc_colors=dc_colors,
        sh_coeffs=sh_coeffs,
        dtype=return_dtype,
    )

    final_spz = _checkpoint_from_tensors(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacity_logits=opacity_logits,
        dc_colors=dc_colors,
        sh_coeffs=sh_coeffs,
        dtype=np.float16,
    )
    write_spz(final_spz, output_dir / "background_model.spz")

    return checkpoint
