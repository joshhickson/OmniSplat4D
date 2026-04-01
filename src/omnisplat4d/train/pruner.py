"""
Phase 1 & 2 — Train: Entropy-constrained opacity pruning.

Two distinct pruning mechanisms are applied during training:

1. Entropy regularisation loss (during gradient updates):
   Adds a penalty term to the training loss that pushes Gaussian opacities
   toward binary states (0 or 1). Semi-transparent floating artifacts are
   driven toward zero opacity, making them candidates for pruning.

2. Hard pruning (periodic, every prune_interval iterations):
   Removes Gaussians whose opacity falls below opacity_prune_threshold.
   These are primitives that the entropy loss has already driven to near-zero.

This combination is called entropy-constrained regularisation and is essential
for memory management on 12GB VRAM — without it, the number of low-opacity
floating Gaussians grows unbounded and eventually OOMs the GPU.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from omnisplat4d.core.types import GaussianCheckpoint

log = logging.getLogger(__name__)


def entropy_regularization_loss(opacities: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy regularization loss over Gaussian opacities.

    The binary entropy function H(p) = -p*log(p) - (1-p)*log(1-p) is maximised
    at p=0.5 and equals zero at p=0 and p=1. By minimising H(sigmoid(opacity)),
    the optimizer is penalised for keeping opacities in ambiguous semi-transparent
    states, driving them toward binary (fully opaque or fully transparent).

    Args:
        opacities: (N, 1) raw (pre-sigmoid) opacity logits from the Gaussian model.

    Returns:
        Scalar loss tensor — mean binary entropy across all Gaussians.
    """
    probs = torch.sigmoid(opacities)
    eps = 1e-6
    probs = probs.clamp(eps, 1.0 - eps)
    entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    return entropy.mean()


def prune_low_opacity(
    checkpoint: GaussianCheckpoint,
    threshold: float,
) -> GaussianCheckpoint:
    """
    Remove Gaussians with opacity below the threshold from a checkpoint.

    Opacities stored in GaussianCheckpoint are post-sigmoid values (already in
    [0, 1]). Primitives with opacity < threshold are deleted by index masking.

    Called every prune_interval iterations during both Phase 1 and Phase 2
    training. After pruning, flush_cuda_cache() should be called by the trainer
    to reclaim the freed GPU memory.

    Args:
        checkpoint: Current GaussianCheckpoint (CPU numpy arrays).
        threshold:  Opacity threshold below which Gaussians are pruned.
                    Default from config: DynamicTrainConfig.opacity_prune_threshold = 0.005.

    Returns:
        New GaussianCheckpoint with low-opacity Gaussians removed.
        The source_path field is cleared (the pruned state is not yet saved).
    """
    opacities_flat = checkpoint.opacities.flatten()
    keep_mask = opacities_flat >= threshold
    n_before = len(opacities_flat)
    n_after = int(keep_mask.sum())
    n_pruned = n_before - n_after
    log.info("Pruning: removed %d / %d Gaussians (threshold=%.4f)", n_pruned, n_before, threshold)

    def _filter(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return arr[keep_mask]

    return GaussianCheckpoint(
        positions=checkpoint.positions[keep_mask],
        rotations=checkpoint.rotations[keep_mask],
        scales=checkpoint.scales[keep_mask],
        opacities=checkpoint.opacities[keep_mask],
        dc_colors=checkpoint.dc_colors[keep_mask],
        sh_coeffs=_filter(checkpoint.sh_coeffs),
        source_path=None,
    )


def densification_mask(
    grad_norms: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Compute a boolean mask of Gaussians that should be densified.

    Densification splits or clones Gaussians whose gradient norm exceeds the
    threshold during the Phase 1 static training adaptive control loop.
    The elevated threshold (0.0004 vs. default 0.0002) is critical for staying
    within the 12GB VRAM budget.

    Args:
        grad_norms: (N,) tensor of position gradient norms per Gaussian.
        threshold:  StaticTrainConfig.densify_grad_threshold (default 0.0004).

    Returns:
        Boolean tensor (N,) — True where densification should occur.
    """
    return grad_norms > threshold
