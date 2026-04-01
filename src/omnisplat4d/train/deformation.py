"""
Phase 2 — Train: Temporal deformation MLP.

Maps a normalised time coordinate t ∈ [0, 1] to per-Gaussian spatial offsets
Δposition (N, 3). The offsets are added to the canonical Gaussian positions
(stored in GaussianCheckpoint.positions) to produce the deformed positions
at time t.

Architecture constraints (ONNX traceability):
    - Fixed-size 3-layer MLP with ReLU activations only.
    - No Python control flow in forward() (no if/else, no variable-length loops).
    - No dynamic tensor shapes — N (number of Gaussians) and hidden_dim are fixed
      at construction time and embedded in the traced graph.
    - Input: scalar t, broadcast to all N Gaussians inside forward().
    - Output: (N, 3) float32 position offsets.

The model is exported to deformation_field.onnx by export/onnx_exporter.py.
At runtime in WebGL/Unity, the ONNX runtime evaluates one forward pass per
frame, passing normalised t for that frame and receiving the offset array.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeformationMLP(nn.Module):
    """
    Fixed-size 3-layer MLP that maps time t to per-Gaussian position offsets.

    Args:
        num_gaussians: Number of Gaussian primitives N (fixed at construction time).
        hidden_dim:    Width of hidden layers (from DynamicTrainConfig.deformation_mlp_hidden_dim).

    Input:
        t: scalar torch.Tensor — normalised time in [0, 1] for the current frame.

    Output:
        offsets: (N, 3) torch.Tensor — position offsets in world space.
    """

    def __init__(self, num_gaussians: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gaussians * 3),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute per-Gaussian position offsets for time t.

        Args:
            t: Scalar tensor (shape [] or [1]) — normalised time ∈ [0, 1].

        Returns:
            offsets: (N, 3) float32 position offsets.
        """
        t_in = t.reshape(1)  # ensure shape (1,)
        flat = self.net(t_in)  # (N*3,)
        return flat.reshape(self.num_gaussians, 3)
