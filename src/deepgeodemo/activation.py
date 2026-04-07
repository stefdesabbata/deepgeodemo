from typing import Any, Literal
import torch
from torch import nn


# Based on TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093
#
# Based on
# https://github.com/openai/sparse_autoencoder
# MIT license

class TopK(nn.Module):
    """TopK activation."""

    def __init__(self,
            postact_fn: nn.Module,
            k: int
            ) -> None:
        super().__init__()
        self.postact_fn = postact_fn
        self.k = k

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If zeroing is disabled, just apply the post-activation function
        output = self.postact_fn(x)
        # Select top-k activations
        topk = torch.topk(output, k=self.k, dim=-1)
        # make all other values 0
        output = torch.zeros_like(output)
        output.scatter_(-1, topk.indices, topk.values)
        return output

# Based on Jumping Ahead's JumpReLU activation function
# by Rajamanoharan et al (2024)
# https://arxiv.org/abs/2407.14435
#
# Based on
# https://github.com/jbloomAus/SAELens/blob/abcf9a603acf9344d249f0a595e89be45b77b7cf/sae_lens/training/training_sae.py#L64
# MIT license
#
# Note: suggested hyperparameters for JumpReLU
# jumprelu_bandwidth=0.001
# jumprelu_init_threshold=0.001

def _rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class _JumpReLUFunction(torch.autograd.Function):
    """
    Implements the JumpReLU activation function from Appendix J of https://arxiv.org/abs/2407.14435
    """
    
    @staticmethod
    def forward(
            x: torch.Tensor,
            threshold: torch.Tensor,
            bandwidth: float,
            ) -> torch.Tensor:
        # Validate threshold tensor
        if not (threshold > 0).all():
            raise ValueError("All values in the threshold tensor must be positive.")
        # Return the JumpReLU activation
        return (x * (x > threshold)).to(x.dtype)

    @staticmethod
    def setup_context(
            ctx:    Any, 
            inputs: tuple[torch.Tensor, torch.Tensor, float], 
            output: torch.Tensor
            ) -> None:
        # Save the input tensors and bandwidth for backward pass
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(
            ctx:         Any, 
            grad_output: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, None]:
        # Retrieve saved tensors and bandwidth
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * grad_output
        # Pseudo-derivative for the threshold using the STE
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None
    
def jump_relu(
        x:         torch.Tensor, 
        threshold: torch.Tensor, 
        bandwidth: float = 0.001
        ) -> torch.Tensor:
    """
    Functional wrapper for the JumpReLU activation function.

    Args:
        x (torch.Tensor): The input tensor (pre-activations).
        threshold (torch.Tensor): The trainable threshold parameter (θ).
        bandwidth (float): The kernel bandwidth hyperparameter (ε).

    Returns:
        torch.Tensor: The output of the JumpReLU activation.
    """
    return _JumpReLUFunction.apply(x, threshold, bandwidth)


class JumpReLU(nn.Module):
    """
    A PyTorch nn.Module for the JumpReLU activation function.

    Args:
        num_features (int): The number of features in the input tensor (e.g., M).
        initial_threshold (float): The initial value for the threshold θ.
        bandwidth (float): The kernel bandwidth hyperparameter ε.
    """
    def __init__(self, 
            num_features:         int, 
            initial_threshold: float = 0.001, 
            bandwidth:         float = 0.001
            ) -> None:
        super().__init__()
        # Bandwidth
        self.bandwidth = bandwidth
        # Threshold
        # To ensure the threshold remains positive, train its logarithm.
        initial_log_threshold = torch.log(torch.tensor(initial_threshold))
        self.log_threshold = nn.Parameter(torch.full((num_features,), initial_log_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the JumpReLU activation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of pre-activations.
        
        Returns:
            The output tensor.
        """
        # Exponentiate to get the positive threshold value for the forward pass.
        threshold = torch.exp(self.log_threshold)
        # Calculate pre-activations (see paper's SAE implementation https://arxiv.org/abs/2407.14435 )
        x = torch.relu(x)
        return jump_relu(x, threshold, self.bandwidth)
