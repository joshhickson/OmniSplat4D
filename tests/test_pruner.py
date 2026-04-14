"""
Tests for src/omnisplat4d/train/pruner.py

No GPU required for most tests — the entropy loss and hard pruning logic
operate on CPU numpy arrays and (for the loss) torch CPU tensors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omnisplat4d.core.types import GaussianCheckpoint
from omnisplat4d.train.pruner import (
    densification_mask,
    entropy_regularization_loss,
    prune_low_opacity,
)


def _make_checkpoint(n: int, opacities: np.ndarray | None = None) -> GaussianCheckpoint:
    """Build a minimal GaussianCheckpoint with N Gaussians for testing."""
    if opacities is None:
        opacities = np.ones((n, 1), dtype=np.float32) * 0.5
    return GaussianCheckpoint(
        positions=np.random.randn(n, 3).astype(np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        scales=np.ones((n, 3), dtype=np.float32),
        opacities=opacities,
        dc_colors=np.random.rand(n, 3).astype(np.float32),
    )


class TestEntropyRegularizationLoss:
    def test_loss_is_scalar(self) -> None:
        opacities = torch.zeros(100, 1)  # all zero logits → sigmoid(0) = 0.5 (max entropy)
        loss = entropy_regularization_loss(opacities)
        assert loss.shape == torch.Size([])

    def test_binary_opacities_have_zero_entropy(self) -> None:
        """Logits of +10 and -10 are effectively 1 and 0 — entropy ≈ 0."""
        opacities = torch.full((100, 1), 10.0)  # sigmoid ≈ 1.0
        loss_high = entropy_regularization_loss(opacities)
        opacities_low = torch.full((100, 1), -10.0)  # sigmoid ≈ 0.0
        loss_low = entropy_regularization_loss(opacities_low)
        assert loss_high.item() < 0.01
        assert loss_low.item() < 0.01

    def test_maximum_entropy_at_zero_logit(self) -> None:
        """sigmoid(0) = 0.5 — maximum binary entropy ≈ ln(2) ≈ 0.693."""
        opacities = torch.zeros(1000, 1)
        loss = entropy_regularization_loss(opacities)
        import math
        assert loss.item() == pytest.approx(math.log(2), abs=0.01)

    def test_loss_is_differentiable(self) -> None:
        opacities = torch.randn(50, 1, requires_grad=True)
        loss = entropy_regularization_loss(opacities)
        loss.backward()
        assert opacities.grad is not None
        assert not torch.isnan(opacities.grad).any()

    def test_loss_decreases_when_pushed_binary(self) -> None:
        """Gradient descent on entropy loss should push logits away from 0.5 probability."""
        # Exact zero logits are a stationary point for entropy under sigmoid parameterization,
        # so we use a tiny perturbation to test the expected descent behavior.
        torch.manual_seed(0)
        opacities = (0.01 * torch.randn(10, 1)).requires_grad_()
        optimizer = torch.optim.SGD([opacities], lr=1.0)
        initial_loss = entropy_regularization_loss(opacities).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = entropy_regularization_loss(opacities)
            loss.backward()
            optimizer.step()
        final_loss = entropy_regularization_loss(opacities).item()
        assert final_loss < initial_loss


class TestPruneLowOpacity:
    def test_removes_correct_gaussians(self) -> None:
        opacities = np.array([[0.001], [0.5], [0.003], [0.8], [0.002]], dtype=np.float32)
        ckpt = _make_checkpoint(5, opacities)
        pruned = prune_low_opacity(ckpt, threshold=0.005)
        # Indices 1 and 3 survive (opacity >= 0.005)
        assert len(pruned.positions) == 2

    def test_all_arrays_have_consistent_length(self) -> None:
        n = 20
        opacities = np.random.rand(n, 1).astype(np.float32)
        ckpt = _make_checkpoint(n, opacities)
        pruned = prune_low_opacity(ckpt, threshold=0.3)
        n_kept = int((opacities >= 0.3).sum())
        assert len(pruned.positions) == n_kept
        assert len(pruned.rotations) == n_kept
        assert len(pruned.scales) == n_kept
        assert len(pruned.opacities) == n_kept
        assert len(pruned.dc_colors) == n_kept

    def test_prune_all_returns_empty(self) -> None:
        opacities = np.zeros((10, 1), dtype=np.float32)
        ckpt = _make_checkpoint(10, opacities)
        pruned = prune_low_opacity(ckpt, threshold=0.01)
        assert len(pruned.positions) == 0

    def test_prune_none_returns_all(self) -> None:
        opacities = np.ones((10, 1), dtype=np.float32)
        ckpt = _make_checkpoint(10, opacities)
        pruned = prune_low_opacity(ckpt, threshold=0.5)
        assert len(pruned.positions) == 10

    def test_source_path_is_cleared(self) -> None:
        from pathlib import Path
        opacities = np.ones((5, 1), dtype=np.float32)
        ckpt = _make_checkpoint(5, opacities)
        ckpt.source_path = Path("some/file.spz")
        pruned = prune_low_opacity(ckpt, threshold=0.5)
        assert pruned.source_path is None

    def test_values_preserved_correctly(self) -> None:
        """Confirm that surviving Gaussians retain their exact original values."""
        opacities = np.array([[0.1], [0.9], [0.01]], dtype=np.float32)
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        ckpt = _make_checkpoint(3, opacities)
        ckpt.positions = positions
        pruned = prune_low_opacity(ckpt, threshold=0.05)
        # Indices 0 and 1 survive
        assert len(pruned.positions) == 2
        assert np.allclose(pruned.positions[0], [1.0, 0.0, 0.0])
        assert np.allclose(pruned.positions[1], [2.0, 0.0, 0.0])


class TestDensificationMask:
    def test_returns_boolean_tensor(self) -> None:
        grad_norms = torch.rand(100)
        mask = densification_mask(grad_norms, threshold=0.5)
        assert mask.dtype == torch.bool

    def test_correct_elements_selected(self) -> None:
        grad_norms = torch.tensor([0.1, 0.6, 0.3, 0.8, 0.0001])
        mask = densification_mask(grad_norms, threshold=0.5)
        expected = torch.tensor([False, True, False, True, False])
        assert torch.all(mask == expected)

    def test_elevated_threshold_reduces_densification(self) -> None:
        """The elevated threshold (0.0004) should select fewer Gaussians than the default (0.0002)."""
        grad_norms = torch.rand(1000) * 0.001  # most will be below 0.0004
        mask_default = densification_mask(grad_norms, threshold=0.0002)
        mask_elevated = densification_mask(grad_norms, threshold=0.0004)
        assert mask_elevated.sum() <= mask_default.sum()
