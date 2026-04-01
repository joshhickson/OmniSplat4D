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
from typing import Optional

from omnisplat4d.core.config import StaticTrainConfig
from omnisplat4d.core.memory import flush_cuda_cache, vram_guard
from omnisplat4d.core.types import GaussianCheckpoint

log = logging.getLogger(__name__)


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
    raise NotImplementedError(
        "train_static: implement gsplat training loop. "
        "See docs/research/phase1_3DGS_static.md for detailed parameter rationale."
    )
