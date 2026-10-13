import pytest
from transformers import LlamaConfig

from elastoformer_moo.arch import ArchInfo, UnsupportedArchitectureError
from elastoformer_moo.config import ElasticConfig, make_grid, smallest_and_largest


def llama_arch(**kw):
    base = dict(
        vocab_size=128, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, head_dim=64,
    )
    return ArchInfo.from_hf_config(LlamaConfig(**{**base, **kw}))


def test_arch_from_config():
    arch = llama_arch()
    assert (arch.num_heads, arch.num_kv_heads, arch.head_dim, arch.group_size) == (32, 8, 64, 4)


def test_head_dim_defaults_to_hidden_over_heads():
    arch = ArchInfo.from_hf_config(LlamaConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=4, intermediate_size=96))
    assert arch.head_dim == 8


@pytest.mark.parametrize("model_type", ["gemma", "phi3", "smollm3", "unknown"])
def test_unsupported_model_type_message(model_type):
    class Cfg:
        pass

    cfg = Cfg()
    cfg.model_type = model_type
    with pytest.raises(UnsupportedArchitectureError, match=model_type):
        ArchInfo.from_hf_config(cfg)


def test_grid_is_sorted_and_ends_at_full_model():
    arch = llama_arch()
    grid = make_grid(arch)
    sizes = [c.size(arch) for c in grid]
    assert sizes == sorted(sizes)
    assert grid[-1].is_full(arch)
    assert all(c.mlp_width % 256 == 0 for c in grid)
    assert [c.kv_groups for c in grid] == [2, 4, 6, 8]


def test_grid_dedups_when_few_kv_groups():
    arch = llama_arch(num_attention_heads=14, num_key_value_heads=2, head_dim=64, intermediate_size=4864)
    grid = make_grid(arch, fractions=(0.1, 0.2, 0.3, 1.0))
    keys = [(c.kv_groups, c.mlp_width) for c in grid]
    assert len(keys) == len(set(keys))
    assert grid[0].kv_groups >= 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(kv_groups=0, mlp_width=256), "kv_groups"),
        (dict(kv_groups=9, mlp_width=256), "kv_groups"),
        (dict(kv_groups=2, mlp_width=0), "mlp_width"),
        (dict(kv_groups=2, mlp_width=9000), "mlp_width"),
        (dict(kv_groups=2, mlp_width=300), "multiple of 256"),
    ],
)
def test_validation_errors_name_the_field(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ElasticConfig(name="x", **kwargs).validate(llama_arch(), mlp_multiple=256)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        make_grid(llama_arch(), fractions=(0.0, 1.0))


def test_smallest_and_largest():
    arch = llama_arch()
    grid = make_grid(arch)
    small, large = smallest_and_largest(list(reversed(grid)), arch)
    assert small == grid[0] and large == grid[-1]
