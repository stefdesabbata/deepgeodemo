import torch
import pytest
from torchgeodemo.autoencoder_search import (
    key_splitter,
    product_dict,
    generate_configs,
    flatten_dict,
)


class TestKeySplitter:

    def test_splits_colon_separated(self):
        assert key_splitter("parent: child") == ["parent", "child"]

    def test_no_colon_returns_single(self):
        assert key_splitter("simple_key") == ["simple_key"]


class TestProductDict:

    def test_basic_product(self):
        results = list(product_dict(a=[1, 2], b=["x", "y"]))
        assert len(results) == 4
        assert {"a": 1, "b": "x"} in results
        assert {"a": 2, "b": "y"} in results

    def test_single_key(self):
        results = list(product_dict(a=[1, 2, 3]))
        assert results == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_empty_values(self):
        results = list(product_dict(a=[]))
        assert results == []


class TestGenerateConfigs:

    def test_correct_count(self):
        base = {"name": "test"}
        options = {"lr": [0.01, 0.001], "depth": [2, 3]}
        configs = generate_configs(base, options)
        assert len(configs) == 4

    def test_nested_key_override(self):
        base = {"autoencoder: latent": 8}
        options = {"autoencoder: depth": [2, 3]}
        configs = generate_configs(base, options)
        assert len(configs) == 2
        for config in configs:
            assert "autoencoder" in config
            assert config["autoencoder"]["latent"] == 8
            assert config["autoencoder"]["depth"] in [2, 3]

    def test_subversioning(self):
        base = {"name": "test"}
        options = {"lr": [0.01, 0.001]}
        configs = generate_configs(base, options, state_subversion_of=5)
        versions = [c["autoencoder"]["version"] for c in configs]
        assert versions == ["5-1", "5-2"]

    def test_no_subversioning(self):
        base = {"name": "test"}
        options = {"lr": [0.01]}
        configs = generate_configs(base, options)
        for config in configs:
            assert "autoencoder" not in config or "version" not in config.get("autoencoder", {})


class TestFlattenDict:

    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"outer": {"inner": 42}}
        assert flatten_dict(d) == {"outer_inner": 42}

    def test_deep_nesting(self):
        d = {"a": {"b": {"c": 1}}}
        assert flatten_dict(d) == {"a_b_c": 1}

    def test_tensor_scalar(self):
        d = {"val": torch.tensor(3.14)}
        result = flatten_dict(d)
        assert pytest.approx(result["val"], abs=1e-4) == 3.14

    def test_non_numeric_becomes_string(self):
        d = {"items": [1, 2, 3]}
        result = flatten_dict(d)
        assert result["items"] == "[1, 2, 3]"
