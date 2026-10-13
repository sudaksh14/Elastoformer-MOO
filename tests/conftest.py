"""Shared helpers: tiny random decoder models and a physical-slice reference oracle."""

from __future__ import annotations

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig

FAMILIES = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "mistral": (MistralConfig, MistralForCausalLM),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
}

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)


def tiny_model(model_type: str, seed: int = 0, **overrides):
    config_cls, model_cls = FAMILIES[model_type]
    torch.manual_seed(seed)
    model = model_cls(config_cls(**{**TINY, **overrides}))
    # Random init leaves biases and norm weights near their defaults; perturb so a bug
    # that slices the wrong rows cannot hide behind identical values.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.02 * torch.randn_like(param))
    return model.eval()


def physical_reference(model, cfg: ElasticConfig):
    """Build a smaller Hugging Face model with the prefix weights physically copied.

    This is an independent oracle: it never touches the elastic forward path.
    """
    arch = ArchInfo.from_hf_config(model.config)
    config_cls, model_cls = FAMILIES[arch.model_type]
    small_cfg = config_cls(
        **{
            **model.config.to_dict(),
            "num_attention_heads": cfg.kv_groups * arch.group_size,
            "num_key_value_heads": cfg.kv_groups,
            "intermediate_size": cfg.mlp_width,
            "head_dim": arch.head_dim,
        }
    )
    small = model_cls(small_cfg).eval()
    # A model built from a constructor can default to a different attention kernel than one
    # loaded with from_pretrained (eager computes softmax in float32; sdpa does not), which
    # makes an "exact" comparison differ at the 1e-3 level on real checkpoints.
    small.config._attn_implementation = model.config._attn_implementation
    q_rows = cfg.kv_groups * arch.group_size * arch.head_dim
    kv_rows = cfg.kv_groups * arch.head_dim
    big, state = model.state_dict(), {}
    for name, tensor in big.items():
        t = tensor
        if "q_proj" in name:
            t = tensor[:q_rows] if name.endswith("weight") or tensor.ndim == 1 else tensor
        elif "k_proj" in name or "v_proj" in name:
            t = tensor[:kv_rows]
        elif "o_proj.weight" in name:
            t = tensor[:, :q_rows]
        elif "gate_proj" in name or "up_proj" in name:
            t = tensor[: cfg.mlp_width]
        elif "down_proj.weight" in name:
            t = tensor[:, : cfg.mlp_width]
        state[name] = t.clone()
    small.load_state_dict(state)
    return small


@pytest.fixture(params=sorted(FAMILIES))
def family(request):
    return request.param


@pytest.fixture
def token_batch():
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 12))
