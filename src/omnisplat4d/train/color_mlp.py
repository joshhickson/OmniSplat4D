"""
Phase 2 — Train: MEGA AC (Alternating Current) color predictor MLP.

Part of the MEGA color decomposition strategy that achieves 48× VRAM compression
compared to spherical harmonics degree 3:

    Full SH Degree 3: 16 coefficients × 3 channels = 48 floats per Gaussian
    MEGA:             3 floats DC + shared MLP weights for AC variations

The AC (view/time-dependent) color component is predicted by a single shared
MLP rather than per-Gaussian SH coefficient arrays. The MLP takes the viewing
direction and normalised time as input and outputs a per-Gaussian color delta
that is added to the DC (base RGB) color.

Architecture constraints (ONNX traceability):
    - Fixed-size 3-layer MLP with ReLU activations only.
    - No Python control flow in forward().
    - Input: (view_dir [3], t [1]) concatenated → (4,).
    - Output: (N, 3) color deltas to add to dc_colors.

The model is exported to ac_color_predictor.onnx by export/onnx_exporter.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ACColorMLP(nn.Module):
    """
    Shared MLP that predicts view/time-dependent color deltas for all Gaussians.

    Instead of per-Gaussian SH coefficient arrays (which scale with N),
    one shared network predicts AC color variations for all N primitives
    simultaneously. This keeps the storage cost independent of N.

    Args:
        num_gaussians: Number of Gaussian primitives N.
        hidden_dim:    Width of hidden layers (from DynamicTrainConfig.ac_mlp_hidden_dim).

    Input:
        view_dir: (3,) float32 — normalised viewing direction in world space.
        t:        scalar float32 — normalised time ∈ [0, 1].

    Output:
        color_delta: (N, 3) float32 — color deltas to add to dc_colors.
    """

    def __init__(self, num_gaussians: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.hidden_dim = hidden_dim

        # Input: view_dir (3) + t (1) = 4 features
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gaussians * 3),
        )

    def forward(self, view_dir: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict AC color deltas for a given viewing direction and time.

        Args:
            view_dir: (3,) normalised viewing direction.
            t:        Scalar tensor — normalised time ∈ [0, 1].

        Returns:
            color_delta: (N, 3) float32 color deltas.
        """
        view_dir_in = view_dir.reshape(3)
        t_in = t.reshape(1)
        x = torch.cat([view_dir_in, t_in], dim=0)  # (4,)
        flat = self.net(x)  # (N*3,)
        return flat.reshape(self.num_gaussians, 3)
