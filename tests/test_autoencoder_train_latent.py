import pytest
from deepgeodemo.autoencoder_train_latent import generate_ae_sizes


class TestGenerateAeSizes:

    def test_endpoints(self):
        sizes = generate_ae_sizes(input_size=100, latent_size=10, depth=3)
        assert sizes[0] == 100
        assert sizes[-1] == 10

    def test_depth_one(self):
        sizes = generate_ae_sizes(input_size=50, latent_size=5, depth=1)
        assert sizes == [50, 5]

    def test_monotonically_decreasing(self):
        sizes = generate_ae_sizes(input_size=200, latent_size=10, depth=5)
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1]

    def test_correct_length(self):
        sizes = generate_ae_sizes(input_size=100, latent_size=10, depth=4)
        assert len(sizes) == 5

    def test_depth_zero_treated_as_one(self):
        sizes = generate_ae_sizes(input_size=50, latent_size=5, depth=0)
        assert sizes == [50, 5]

    def test_equal_input_and_latent(self):
        sizes = generate_ae_sizes(input_size=10, latent_size=10, depth=3)
        assert sizes[0] == 10
        assert sizes[-1] == 10
        assert all(s == 10 for s in sizes)
