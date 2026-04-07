# Libraries
from typing import Literal
import math
import numpy as np
import torch
from torch import nn
import lightning as L
import copy

# This project's modules
from .activation import TopK, JumpReLU
from .loss import normalized_mean_squared_error, mean_absolute_error, normalized_L1_loss, normalized_L0_loss, topk_aux_loss


class MLP(L.LightningModule):
    """
    Multi-Layer Perceptron (MLP) model implemented using PyTorch Lightning.

    Args:
        size_sequence (list[int]): A list of integers specifying the size of each layer in the MLP.
        use_batch_norm (bool, optional): Whether to use batch normalization after each linear layer. Defaults to False.
        negative_slope (float, optional): The negative slope value for the LeakyReLU activation function. Defaults to 0.01.

    Attributes:
        size_sequence (list[int]): The size of each layer in the MLP.
        mlp (torch.nn.Sequential): The sequential container holding the MLP layers.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            MLP forward pass. Takes an input tensor `x` and returns the output tensor after passing through the MLP.
    """

    def __init__(self, 
            size_sequence:  list[int], 
            use_batch_norm: bool  = False,
            negative_slope: float = 0.01
            ) -> None:
        super(MLP, self).__init__()
        # Set parameters
        self.size_sequence = size_sequence
        self.use_batch_norm = use_batch_norm

        # Create Multi-Layer Perceptron based on size sequence
        self.mlp = torch.nn.Sequential()
        mlp_layers = len(self.size_sequence) - 1
        # Add all layers but the last
        for i in range(mlp_layers):
            self.mlp.append(
                torch.nn.Linear(self.size_sequence[i], self.size_sequence[i + 1]))
            # Add activation function and batch normalization except for the last layer
            if i < mlp_layers - 1:
                # Add batch normalization if specified
                if self.use_batch_norm:
                    self.mlp.append(
                        torch.nn.BatchNorm1d(self.size_sequence[i + 1]))
                # Add activation function 
                self.mlp.append(
                    torch.nn.LeakyReLU(negative_slope=negative_slope))
        # # Add final layer
        # self.mlp.append(
        #     torch.nn.Linear(self.size_sequence[-2], self.size_sequence[-1]))
        # if self.use_batch_norm:
        #     self.mlp.append(
        #         torch.nn.BatchNorm1d(self.size_sequence[-1]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AutoEncoder(L.LightningModule):
    """
    AutoEncoder implemented in PyTorch Lightning.

    Args:
        encoder_sizes (list[int]): List of integers specifying the sizes of the encoder layers.
        encoder_sparse (bool, optional): If True, creates a sparse encoder. Default is False.
        encoder_sparse_topk_k (int, optional): If encoder_sparse is True, specifies the top-k sparsity constraint. Default is None.
        encoder_activation (Literal["Identity", "ReLU", "Tanh", "Sigmoid"], optional): Final activation function for the encoder. Default is "Identity".
        decoder_sizes (list[int], optional): List of integers specifying the sizes of the decoder layers. If None, the decoder sizes are set to be the reverse of encoder sizes. Default is None.
        decoder_activation (Literal["Identity", "ReLU", "Tanh", "Sigmoid"], optional): Final activation function for the decoder. Default is "Identity".
        use_batch_norm (bool, optional): If True, adds batch normalization layers. Default is False.
        loss_weight_latent_l0 (float, optional): Weight for the latent L0 loss. Default is 0.0.
        loss_weight_latent_l1 (float, optional): Weight for the latent L1 loss if a sparse autoencoder is created. Default is 0.01.
        loss_weight_covariance (float, optional): Weight for the covariance loss. Default is 0.0.
        loss_weight_auxk (float, optional): Weight for the auxiliary TopK loss if a sparse autoencoder is created. Default is 0.0.
        regu_weight_l2 (float, optional): Weight for L2 regularization. Default is 0.0.
        regu_weight_l1 (float, optional): Weight for L1 regularization. Default is 0.0.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        patience (int, optional): Patience for the learning rate scheduler
        verbose (bool, optional): If True, prints additional information. Default is False.

    Methods:
        forward(x):
            Forward pass through the autoencoder. Returns embeddings and reconstruction.
        
        encode(x):
            Encodes the input data and returns the embeddings.
        
        training_step(batch, batch_idx):
            Defines the training step. Computes the loss and logs it.
        
        configure_optimizers():
            Configures the optimizer and learning rate scheduler.
    """

    def __init__(self, 
            encoder_sizes:       list[int], 
            encoder_sparse:          bool = False,
            encoder_sparse_topk_k:    int = None,
            encoder_activation:    Literal["Identity", "JumpReLU", "LeakyReLU", "ReLU", "Tanh", "Sigmoid"] = "Identity",
            decoder_sizes:      list[int] = None, 
            decoder_activation:    Literal["Identity", "JumpReLU", "LeakyReLU", "ReLU", "Tanh", "Sigmoid"] = "Identity", 
            use_batch_norm:          bool = False,
            loss_weight_latent_l0:  float = 0.0,
            loss_weight_latent_l1:  float = 0.0,
            loss_weight_covariance: float = 0.0,
            loss_weight_auxk:       float = 0.0,
            regu_weight_l2:         float = 0.0,
            regu_weight_l1:         float = 0.0,
            learning_rate:          float = 1e-3,
            patience:                 int = 10,
            verbose:                 bool = False
            ) -> None:
        super(AutoEncoder, self).__init__()

        # Set parameters
        self.encoder_sizes = encoder_sizes
        self.encoder_sparse = encoder_sparse
        self.encoder_sparse_topk_k = encoder_sparse_topk_k
        self.encoder_activation_type = encoder_activation
        self.decoder_sizes = decoder_sizes
        self.decoder_activation_type = decoder_activation
        self.use_batch_norm = use_batch_norm
        self.loss_weight_latent_l0 = loss_weight_latent_l0
        self.loss_weight_latent_l1 = loss_weight_latent_l1
        self.loss_weight_covariance = loss_weight_covariance
        self.loss_weight_auxk = loss_weight_auxk
        self.regu_weight_l2 = regu_weight_l2
        self.regu_weight_l1 = regu_weight_l1
        self.learning_rate = learning_rate
        self.patience = patience
        
        # Encoder
        self.encoder = MLP(
            self.encoder_sizes, 
            self.use_batch_norm)
        
        # Add final activation
        self.encoder_preactivation = nn.Identity()
        self.encoder_activation    = nn.Identity()
        # If specified, add TopK activation
        if self.encoder_sparse:
            self.encoder_preactivation = nn.BatchNorm1d(self.encoder_sizes[-1])
            # self.encoder_preactivation.weight.data.fill_(0.5)
            self.encoder_preactivation.bias.data.fill_(1.0)
            # If not specified, use half of the output size
            if self.encoder_sparse_topk_k is None:
                self.encoder_sparse_topk_k = math.floor(self.encoder_sizes[-1] / 2)
            # Create base activation
            if self.encoder_activation_type   ==  'JumpReLU':
                encoder_activation_base  =    JumpReLU(self.encoder_sizes[-1])
            elif self.encoder_activation_type ==  'LeakyReLU':
                encoder_activation_base  =    nn.LeakyReLU()
            elif self.encoder_activation_type ==  'ReLU':
                encoder_activation_base  =    nn.ReLU()
            elif self.encoder_activation_type ==  'Tanh':
                encoder_activation_base  =    nn.Tanh()
            elif self.encoder_activation_type ==  'Sigmoid':
                encoder_activation_base  =    nn.Sigmoid()
            else:
                encoder_activation_base  = nn.Identity()
            # Create TopK activation
            self.encoder_activation = TopK(
                encoder_activation_base,
                self.encoder_sparse_topk_k)
        # Otherwise, add standard activation
        else:
            if self.encoder_activation_type   ==  'JumpReLU':
                self.encoder_activation  =    JumpReLU(self.encoder_sizes[-1])
            elif self.encoder_activation_type ==  'LeakyReLU':
                self.encoder_activation  =    nn.LeakyReLU()
            elif self.encoder_activation_type ==  'ReLU':
                self.encoder_activation  =    nn.ReLU()
            elif self.encoder_activation_type ==  'Tanh':
                self.encoder_activation  =    nn.Tanh()
            elif self.encoder_activation_type ==  'Sigmoid':
                self.encoder_activation  =    nn.Sigmoid()
        # Keep track of active neurons for sparse autoencoders
        self.active_neurons_train = None
        self.active_neurons_val = None
        
        # Decoder
        if decoder_sizes is None:
            # If not specified, assume symmetrical autoencoder and use reverse of encoder sizes
            self.decoder_sizes = list(reversed(encoder_sizes))
            print(f"Decoder sizes not specified. Using reverse of encoder sizes: {self.decoder_sizes}.") if verbose else None
        self.decoder = MLP(
            self.decoder_sizes, 
            self.use_batch_norm)
        
        self.decoder_activation      =   nn.Identity()
        if self.decoder_activation_type   ==  'JumpReLU':
            self.decoder_activation  =    JumpReLU(self.decoder_sizes[-1])
        elif self.decoder_activation_type ==  'LeakyReLU':
            self.decoder_activation  = nn.LeakyReLU()
        elif self.decoder_activation_type ==  'ReLU':
            self.decoder_activation  = nn.ReLU()
        elif self.decoder_activation_type ==  'Tanh':
            self.decoder_activation  = nn.Tanh()
        elif self.decoder_activation_type ==  'Sigmoid':
            self.decoder_activation  = nn.Sigmoid()
    
    
    # Forward pass
    def forward(self, x):
        pre_embeddings = self.encoder_preactivation(self.encoder(x))
        embeddings     = self.encoder_activation(pre_embeddings)
        reconstruction = self.decoder_activation(self.decoder(embeddings))
        return pre_embeddings, embeddings, reconstruction
    
    # Encode and decode function
    def encode(self, x):
        return self.encoder_activation(self.encoder_preactivation(self.encoder(x)))
    def decode(self, x):
        return self.decoder_activation(self.decoder(x))
    
    # Training step
    def training_step(self, batch, batch_idx):
        return self._a_step(batch, batch_idx, log_prefix='train_')
    
    # Validation step
    def validation_step(self, batch, batch_idx):
        return self._a_step(batch, batch_idx, log_prefix='val_')


    # A step
    def _a_step(self, batch, batch_idx, log_prefix: str = ''):
        
        # Forward pass
        pre_embeddings, embeddings, reconstruction = self.forward(batch)

        # MAE loss for check
        mae = mean_absolute_error(reconstruction, batch)
        self.log(f'{log_prefix}MAE', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Reconstruction loss
        loss = normalized_mean_squared_error(reconstruction, batch)
        self.log(f'{log_prefix}recon_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Covariance loss
        # crate centered embeddings
        batch_size = embeddings.size(0)
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        # calculate ovariance matrix
        cov_matrix = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
        # sum of squares of elements, excluding the diagonal
        covariance_loss = (cov_matrix * (
                1 - torch.eye(cov_matrix.size(0), device=embeddings.device))
            ).abs().sum()
        self.log(f'{log_prefix}covariance_loss', self.loss_weight_covariance * covariance_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        loss += self.loss_weight_covariance * covariance_loss

        if self.encoder_sparse:

            # Track active neurons
            if log_prefix == 'train_':
                self.active_neurons_train += embeddings.sum(dim=0)
            if log_prefix == 'val_':
                self.active_neurons_val   += embeddings.sum(dim=0)

            # Log how many embedding dimensions are effectively zero
            emb_fired = (embeddings>0).float().sum(dim=0)
            emb_dead  = emb_fired == 0

            # Auxiliary TopK loss (avoid dead neurons)
            auxk_embeddings = torch.where(emb_dead, pre_embeddings, 0.0)
            auxk_embeddings_topk = torch.topk(auxk_embeddings, k=self.encoder_sparse_topk_k, dim=-1)
            auxk_embeddings = torch.zeros_like(auxk_embeddings)
            auxk_embeddings.scatter_(-1, auxk_embeddings_topk.indices, auxk_embeddings_topk.values)
            auxk_reconstruction = self.decode(auxk_embeddings)
            auxk_loss = topk_aux_loss(batch, emb_dead, auxk_reconstruction, reconstruction)
            self.log(f'{log_prefix}auxk_loss', self.loss_weight_auxk * auxk_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            loss += self.loss_weight_auxk * auxk_loss

            # Latent L0 or L1 loss
            if self.decoder_activation_type == 'JumpReLU':
                latent_l0_loss = normalized_L0_loss(embeddings)
                self.log(f'{log_prefix}latent_l0_loss', self.loss_weight_latent_l0 * latent_l0_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                loss += self.loss_weight_latent_l0 * latent_l0_loss
            else:
                latent_l1_loss = normalized_L1_loss(embeddings, batch)
                self.log(f'{log_prefix}latent_l1_loss', self.loss_weight_latent_l1 * latent_l1_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                loss += self.loss_weight_latent_l1 * latent_l1_loss
        
        # L1 weight regularisation
        if self.regu_weight_l1 > 0.0:
            l1_reg = torch.tensor(0.0, device=batch.device)
            for p in self.parameters():
                l1_reg += p.abs().sum()
            self.log(f'{log_prefix}l1_reg_loss', self.regu_weight_l1 * l1_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            loss += self.regu_weight_l1 * l1_reg
    
        self.log(f'{log_prefix}loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Return loss
        return loss
    
    def on_train_epoch_start(self):
        # Keep track of active / dead neurons for sparse autoencoders
        if self.encoder_sparse:
            self.active_neurons_train = torch.zeros(self.encoder_sizes[-1], dtype=torch.float32, device=self.device)
    def on_validation_epoch_start(self):
        # Keep track of active / dead neurons for sparse autoencoders
        if self.encoder_sparse:
            self.active_neurons_val   = torch.zeros(self.encoder_sizes[-1], dtype=torch.float32, device=self.device)
    def on_train_epoch_end(self):
        # Keep track of active / dead neurons for sparse autoencoders
        if self.encoder_sparse:
            emb_dead_train  = (self.active_neurons_train == 0).float().sum() / self.encoder_sizes[-1]
            self.log(f'train_emb_dead', emb_dead_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def on_validation_epoch_end(self):
        # Keep track of active / dead neurons for sparse autoencoders
        if self.encoder_sparse:
            emb_dead_val    = (self.active_neurons_val   == 0).float().sum() / self.encoder_sizes[-1]
            self.log(f'val_emb_dead', emb_dead_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    # Optimizer details
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.regu_weight_l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=self.patience, min_lr=1e-10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}
