import torch
import pytest
from deepgeodemo.loss import (
    normalized_mean_squared_error,
    mean_absolute_error,
    normalized_L1_loss,
    normalized_L0_loss,
    topk_aux_loss,
)


class TestNormalizedMSE:

    def test_identical_inputs_return_zero(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert normalized_mean_squared_error(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        original = torch.tensor([[1.0, 0.0]])
        reconstruction = torch.tensor([[0.0, 0.0]])
        # numerator mean = 0.5, denominator = 0.5, ratio = 1.0
        result = normalized_mean_squared_error(reconstruction, original)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_returns_scalar(self):
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        result = normalized_mean_squared_error(x, y)
        assert result.dim() == 0


class TestMAE:

    def test_identical_inputs_return_zero(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert mean_absolute_error(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[1.0, 3.0]])
        # (1 + 3) / 2 = 2
        assert mean_absolute_error(a, b).item() == pytest.approx(2.0, abs=1e-5)


class TestNormalizedL1Loss:

    def test_zero_latent(self):
        latent = torch.zeros(4, 8)
        original = torch.randn(4, 8)
        assert normalized_L1_loss(latent, original).item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_nonzero_latent(self):
        latent = torch.ones(4, 8)
        original = torch.ones(4, 8)
        assert normalized_L1_loss(latent, original).item() > 0


class TestNormalizedL0Loss:

    def test_zero_latent(self):
        latent = torch.zeros(4, 8)
        assert normalized_L0_loss(latent).item() == pytest.approx(0.0, abs=1e-7)

    def test_all_nonzero_latent(self):
        latent = torch.ones(4, 8)
        assert normalized_L0_loss(latent).item() == pytest.approx(1.0, abs=1e-7)

    def test_half_nonzero(self):
        latent = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        assert normalized_L0_loss(latent).item() == pytest.approx(0.5, abs=1e-7)


class TestTopkAuxLoss:

    def test_no_dead_neurons_returns_zero(self):
        batch = torch.randn(4, 8)
        reconstruction = torch.randn(4, 8)
        auxk_reconstruction = torch.randn(4, 8)
        neurons_dead = torch.zeros(8, dtype=torch.bool)
        result = topk_aux_loss(batch, neurons_dead, auxk_reconstruction, reconstruction)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_none_neurons_dead_returns_zero(self):
        batch = torch.randn(4, 8)
        reconstruction = torch.randn(4, 8)
        auxk_reconstruction = torch.randn(4, 8)
        neurons_dead = None
        result = topk_aux_loss(batch, neurons_dead, auxk_reconstruction, reconstruction)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_with_some_dead_neurons_returns_positive(self):
        batch = torch.randn(4, 8)
        reconstruction = torch.zeros(4, 8)
        auxk_reconstruction = torch.randn(4, 8)
        neurons_dead = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.bool)
        result = topk_aux_loss(batch, neurons_dead, auxk_reconstruction, reconstruction)
        assert result.item() > 0

    def test_with_all_dead_neurons_returns_positive(self):
        batch = torch.randn(4, 8)
        reconstruction = torch.zeros(4, 8)
        auxk_reconstruction = torch.randn(4, 8)
        neurons_dead = torch.ones(8, dtype=torch.bool)
        result = topk_aux_loss(batch, neurons_dead, auxk_reconstruction, reconstruction)
        assert result.item() > 0
