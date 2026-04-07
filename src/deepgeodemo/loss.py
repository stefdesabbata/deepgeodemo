import torch
from torch import nn


# Loss functions based on
# https://github.com/openai/sparse_autoencoder
# MIT license

def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    # print(reconstruction)
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / ((original_input**2).mean(dim=1) + 1e-8)
    ).mean()

def mean_absolute_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: mean absolute error (shape: [1])
    """
    # print(reconstruction)
    return (
        ((reconstruction - original_input).abs()).mean(dim=1)
    ).mean()


# Loss funcrions for TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093

def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()


# Based on Jumping Ahead's JumpReLU activation function
# by Rajamanoharan et al (2024)
# https://arxiv.org/abs/2407.14435

def normalized_L0_loss(
    latent_activations: torch.Tensor,
) -> torch.Tensor:
    return (latent_activations > 0).to(latent_activations.dtype).sum(dim=1).mean() / latent_activations.shape[1]


# Based on TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093
#
# Based on
# https://github.com/jbloomAus/SAELens/blob/abcf9a603acf9344d249f0a595e89be45b77b7cf/sae_lens/training/training_sae.py#L466
# MIT license

def topk_aux_loss(
        batch: torch.Tensor,
        neurons_dead: torch.Tensor,
        auxk_reconstruction: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
    if neurons_dead is None or (num_dead := int(neurons_dead.sum())) == 0:
        return reconstruction.new_tensor(0.0)
    
    # Calculate residuals
    residual = (batch - reconstruction).detach()

    # Heuristic from Appendix B.1 in the paper
    k_aux = batch.shape[-1] // 2
    # Reduce the scale of the loss if there are a small number of dead latents
    scale = min(num_dead / k_aux, 1.0)
    k_aux = min(k_aux, num_dead)

    # auxk_loss = (auxk_reconstruction - residual).pow(2).sum(dim=-1).mean()
    auxk_loss = normalized_mean_squared_error(auxk_reconstruction, residual)
    return scale * auxk_loss

