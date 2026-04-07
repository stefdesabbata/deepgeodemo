import torch
import pytest
from torchgeodemo.activation import TopK, JumpReLU, _rectangle


class TestTopK:

    def test_exactly_k_nonzero_per_sample(self):
        topk = TopK(postact_fn=torch.nn.Identity(), k=3)
        x = torch.randn(5, 10)
        out = topk(x)
        nonzero_counts = (out != 0).sum(dim=-1)
        assert (nonzero_counts == 3).all()

    def test_post_activation_applied(self):
        """TopK should apply post-activation to kept values."""
        topk = TopK(postact_fn=torch.nn.ReLU(), k=5)
        # Input with mix of positive and negative values
        x = torch.tensor([[3.0, -2.0, 1.0, -4.0, 5.0, -1.0, 2.0, -3.0, 4.0, -5.0]])
        out = topk(x)
        # All kept values should be >= 0 (ReLU applied)
        assert (out >= 0).all()

    def test_gradient_flows(self):
        topk = TopK(postact_fn=torch.nn.Identity(), k=3)
        x = torch.randn(4, 8, requires_grad=True)
        out = topk(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Gradients should be non-zero only for the top-k positions
        assert (x.grad != 0).sum(dim=-1).tolist() == [3, 3, 3, 3]

    def test_k_equals_dim_keeps_all(self):
        topk = TopK(postact_fn=torch.nn.Identity(), k=4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = topk(x)
        assert torch.allclose(out, x)


class TestJumpReLU:

    def test_zeros_below_threshold(self):
        jr = JumpReLU(num_features=4, initial_threshold=0.5)
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        out = jr(x)
        assert (out == 0).all()

    def test_passes_above_threshold(self):
        jr = JumpReLU(num_features=4, initial_threshold=0.1)
        x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        out = jr(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_negative_inputs_zeroed(self):
        jr = JumpReLU(num_features=3, initial_threshold=0.001)
        x = torch.tensor([[-5.0, -1.0, -0.1]])
        out = jr(x)
        assert (out == 0).all()

    def test_threshold_is_learnable(self):
        jr = JumpReLU(num_features=4, initial_threshold=0.5, bandwidth=10.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = jr(x)
        loss = out.sum()
        loss.backward()
        assert jr.log_threshold.grad is not None
        assert jr.log_threshold.grad.abs().sum() > 0


class TestRectangle:

    def test_inside_bandwidth(self):
        x = torch.tensor([0.0, 0.1, -0.1, 0.49])
        assert (_rectangle(x) == 1).all()

    def test_outside_bandwidth(self):
        x = torch.tensor([-1.0, 1.0, 0.51, -0.51])
        assert (_rectangle(x) == 0).all()

    def test_at_boundary(self):
        x = torch.tensor([-0.5, 0.5])
        assert (_rectangle(x) == 0).all()
