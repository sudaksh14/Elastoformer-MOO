"""Materialize one nested configuration as a standalone, smaller Hugging Face model.

The result copies the prefix weights into a physically smaller model. It is used for two
things: as an independent oracle in tests (it never touches the elastic forward path), and as
the "independently stored checkpoint" baseline when comparing against shared-weight switching.
"""

from __future__ import annotations

import torch

from .arch import ArchInfo
from .config import ElasticConfig


@torch.no_grad()
def materialize(model, cfg: ElasticConfig):
    """Return a new model of the same class holding exactly the weights ``cfg`` uses."""
    arch = ArchInfo.from_hf_config(model.config)
    cfg.validate(arch)
    small_cfg = type(model.config)(
        **{
            **model.config.to_dict(),
            "num_attention_heads": cfg.kv_groups * arch.group_size,
            "num_key_value_heads": cfg.kv_groups,
            "intermediate_size": cfg.mlp_width,
            "head_dim": arch.head_dim,
        }
    )
    small = type(model)(small_cfg).to(next(model.parameters()).dtype).eval()
    # A model built from a constructor can default to a different attention kernel than one
    # loaded with from_pretrained (eager computes softmax in float32; sdpa does not).
    small.config._attn_implementation = model.config._attn_implementation

    q_rows = cfg.kv_groups * arch.group_size * arch.head_dim
    kv_rows = cfg.kv_groups * arch.head_dim
    state = {}
    for name, tensor in model.state_dict().items():
        if "q_proj" in name:
            tensor = tensor[:q_rows]
        elif "k_proj" in name or "v_proj" in name:
            tensor = tensor[:kv_rows]
        elif "o_proj.weight" in name:
            tensor = tensor[:, :q_rows]
        elif "gate_proj" in name or "up_proj" in name:
            tensor = tensor[: cfg.mlp_width]
        elif "down_proj.weight" in name:
            tensor = tensor[:, : cfg.mlp_width]
        state[name] = tensor.clone()
    small.load_state_dict(state)
    return small
