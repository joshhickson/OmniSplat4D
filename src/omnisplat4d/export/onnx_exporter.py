"""
Phase 4 — Export: ONNX export for DeformationMLP and ACColorMLP.

Exports the trained MLP models to ONNX format for use in WebGL and Unity
streaming pipelines. The ONNX runtime evaluates these models at rendering
time to compute per-frame Gaussian deformations and color variations.

ONNX compatibility requirements:
    - Both MLPs use only fixed-size Linear layers and ReLU activations.
    - No Python control flow in forward() — torch.onnx.export() traces cleanly.
    - Opset 17 (from ExportConfig.onnx_opset) is the minimum for compatibility
      with web-based ONNX runtimes.

Output files per temporal chunk:
    deformation_field.onnx   — maps t (scalar) → position offsets (N, 3)
    ac_color_predictor.onnx  — maps (view_dir [3], t [1]) → color deltas (N, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from omnisplat4d.core.config import ExportConfig
from omnisplat4d.train.color_mlp import ACColorMLP
from omnisplat4d.train.deformation import DeformationMLP

log = logging.getLogger(__name__)


def export_deformation_mlp(
    model: DeformationMLP,
    output_path: Path,
    cfg: ExportConfig,
) -> None:
    """
    Export a trained DeformationMLP to ONNX format.

    Uses torch.onnx.export() with a dummy input to trace the computation graph.
    The exported model accepts a scalar t and outputs (N, 3) position offsets.

    Args:
        model:       Trained DeformationMLP instance (eval mode will be set).
        output_path: Destination .onnx file path.
        cfg:         ExportConfig (provides onnx_opset).

    Side effects:
        Creates parent directories if they do not exist.
        Sets model to eval mode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_t = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy_t,),
            f=str(output_path),
            input_names=["t"],
            output_names=["position_offsets"],
            opset_version=cfg.onnx_opset,
            do_constant_folding=True,
        )
    size_kb = output_path.stat().st_size / 1024
    log.info("Exported DeformationMLP → %s (%.1f KB)", output_path, size_kb)


def export_color_mlp(
    model: ACColorMLP,
    output_path: Path,
    cfg: ExportConfig,
) -> None:
    """
    Export a trained ACColorMLP to ONNX format.

    The exported model accepts (view_dir [3], t [1]) and outputs (N, 3) color
    deltas to be added to dc_colors at render time.

    Args:
        model:       Trained ACColorMLP instance (eval mode will be set).
        output_path: Destination .onnx file path.
        cfg:         ExportConfig (provides onnx_opset).

    Side effects:
        Creates parent directories if they do not exist.
        Sets model to eval mode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_view_dir = torch.zeros(3, dtype=torch.float32)
    dummy_t = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy_view_dir, dummy_t),
            f=str(output_path),
            input_names=["view_dir", "t"],
            output_names=["color_delta"],
            opset_version=cfg.onnx_opset,
            do_constant_folding=True,
        )
    size_kb = output_path.stat().st_size / 1024
    log.info("Exported ACColorMLP → %s (%.1f KB)", output_path, size_kb)
