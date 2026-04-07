import torch
import pytest
from deepgeodemo.models import MLP, AutoEncoder


class TestMLP:

    def test_output_shape(self):
        mlp = MLP(size_sequence=[10, 8, 4])
        x = torch.randn(5, 10)
        out = mlp(x)
        assert out.shape == (5, 4)
        assert mlp.mlp.__len__() == 3

    def test_batch_norm_layers_present(self):
        mlp = MLP(size_sequence=[10, 8, 4], use_batch_norm=True)
        layer_types = [type(m).__name__ for m in mlp.mlp]
        assert "BatchNorm1d" in layer_types
        assert mlp.mlp.__len__() == 4

    def test_batch_norm_layers_absent(self):
        mlp = MLP(size_sequence=[10, 8, 4], use_batch_norm=False)
        layer_types = [type(m).__name__ for m in mlp.mlp]
        assert "BatchNorm1d" not in layer_types
        assert mlp.mlp.__len__() == 3

    def test_single_layer(self):
        mlp = MLP(size_sequence=[6, 3])
        x = torch.randn(2, 6)
        out = mlp(x)
        assert out.shape == (2, 3)
        assert mlp.mlp.__len__() == 1



class TestAutoEncoder:

    def test_forward_returns_three_tensors(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4])
        ae.eval()
        x = torch.randn(8, 10)
        pre_emb, emb, recon = ae(x)
        assert pre_emb.shape == (8, 4)
        assert emb.shape == (8, 4)
        assert recon.shape == (8, 10)

    def test_encode_output_shape(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4])
        ae.eval()
        x = torch.randn(8, 10)
        emb = ae.encode(x)
        assert emb.shape == (8, 4)

    def test_decode_output_shape(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4])
        ae.eval()
        latent = torch.randn(8, 4)
        recon = ae.decode(latent)
        assert recon.shape == (8, 10)

    def test_symmetric_decoder_default(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4])
        assert ae.decoder_sizes == [4, 6, 10]

    def test_custom_decoder_sizes(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4], decoder_sizes=[4, 8, 10])
        assert ae.decoder_sizes == [4, 8, 10]

    def test_sparse_topk_produces_sparse_latent(self):
        ae = AutoEncoder(
            encoder_sizes=[10, 6, 8],
            encoder_sparse=True,
            encoder_sparse_topk_k=3,
        )
        ae.eval()
        x = torch.randn(4, 10)
        emb = ae.encode(x)
        # At most k values should be non-zero per sample
        nonzero_per_sample = (emb != 0).sum(dim=-1)
        assert (nonzero_per_sample <= 3).all()

    def test_encoder_activation_relu(self):
        ae = AutoEncoder(
            encoder_sizes=[10, 6, 4],
            encoder_activation="ReLU",
        )
        ae.eval()
        x = torch.randn(4, 10)
        emb = ae.encode(x)
        assert (emb >= 0).all()

    def test_batch_norm_in_autoencoder(self):
        ae = AutoEncoder(encoder_sizes=[10, 6, 4], use_batch_norm=True)
        layer_types = [type(m).__name__ for m in ae.encoder.mlp]
        assert "BatchNorm1d" in layer_types
