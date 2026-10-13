"""Run one set of weights at several nested operating points.

A configuration keeps the first ``kv_groups`` key-value groups of attention and the
first ``mlp_width`` channels of every MLP. Weights are never copied: the elastic
forward slices the shared tensors with views, so all configurations train and run on
the same parameters.

The wrappers patch ``forward`` on the existing attention and MLP modules instead of
replacing them. Parameter names, ``state_dict`` keys and ``save_pretrained`` output are
therefore identical to the original Hugging Face model.

Supported blocks: separate q/k/v/o and gate/up/down projections (llama, mistral, qwen2,
qwen3). Grouped-query attention is handled by keeping whole KV groups, so head
dimension, RoPE and the KV-cache layout are unchanged.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .arch import ArchInfo
from .config import ElasticConfig


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return F.linear(x, weight, bias)


def _sliding_window(attn) -> Optional[int]:
    """Sliding-window size for this layer, following the Hugging Face convention."""
    if getattr(attn, "sliding_window", None) is not None:
        return attn.sliding_window
    cfg = attn.config
    if getattr(cfg, "model_type", None) == "mistral":
        return getattr(cfg, "sliding_window", None)
    if (
        getattr(cfg, "use_sliding_window", False)
        and getattr(cfg, "sliding_window", None) is not None
        and attn.layer_idx >= getattr(cfg, "max_window_layers", 0)
    ):
        return cfg.sliding_window
    return None


def _elastic_attention(
    attn,
    arch: ArchInfo,
    cfg: ElasticConfig,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask=None,
    past_key_value=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Attention over the first ``cfg.kv_groups`` KV groups and their query heads."""
    module = sys.modules[type(attn).__module__]
    groups, group_size, head_dim = cfg.kv_groups, arch.group_size, arch.head_dim
    q_rows, kv_rows = groups * group_size * head_dim, groups * head_dim
    batch, seq, _ = hidden_states.shape

    def project(proj, rows):
        bias = None if proj.bias is None else proj.bias[:rows]
        return _linear(hidden_states, proj.weight[:rows], bias)

    q = project(attn.q_proj, q_rows).view(batch, seq, groups * group_size, head_dim)
    k = project(attn.k_proj, kv_rows).view(batch, seq, groups, head_dim)
    v = project(attn.v_proj, kv_rows).view(batch, seq, groups, head_dim)
    if hasattr(attn, "q_norm"):  # qwen3: RMSNorm over head_dim, shared by all heads
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    cos, sin = position_embeddings
    q, k = module.apply_rotary_pos_emb(q, k, cos, sin)

    cache = past_key_values if past_key_values is not None else past_key_value
    if cache is not None:
        k, v = cache.update(k, v, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    interface = module.eager_attention_forward
    impl = attn.config._attn_implementation
    if impl != "eager":
        interface = ALL_ATTENTION_FUNCTIONS[impl]
    kwargs.pop("sliding_window", None)
    out, _ = interface(
        attn,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not attn.training else getattr(attn, "attention_dropout", 0.0),
        scaling=getattr(attn, "scaling", head_dim**-0.5),
        sliding_window=_sliding_window(attn),
        **kwargs,
    )
    out = out.reshape(batch, seq, q_rows).contiguous()
    return _linear(out, attn.o_proj.weight[:, :q_rows], attn.o_proj.bias), None


def _elastic_mlp(mlp, cfg: ElasticConfig, x: torch.Tensor) -> torch.Tensor:
    """Gated MLP over the first ``cfg.mlp_width`` intermediate channels."""
    width = cfg.mlp_width
    gate = _linear(x, mlp.gate_proj.weight[:width], None if mlp.gate_proj.bias is None else mlp.gate_proj.bias[:width])
    up = _linear(x, mlp.up_proj.weight[:width], None if mlp.up_proj.bias is None else mlp.up_proj.bias[:width])
    return _linear(mlp.act_fn(gate) * up, mlp.down_proj.weight[:, :width], mlp.down_proj.bias)


def _decoder_layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(f"{type(model).__name__} has no .model.layers; cannot attach elastic wrappers")
    return layers


class ElasticController:
    """Attach to a Hugging Face causal LM and switch its operating point.

    ``controller.cfg`` is the active configuration; ``None`` means the full model.
    """

    def __init__(self, model, arch: Optional[ArchInfo] = None, mlp_multiple: int = 1):
        self.model = model
        self.arch = arch or ArchInfo.from_hf_config(model.config)
        self.mlp_multiple = mlp_multiple
        self.cfg: Optional[ElasticConfig] = None
        self._attached = False

    def attach(self) -> "ElasticController":
        if self._attached:
            return self
        for layer in _decoder_layers(self.model):
            self._patch_attention(layer.self_attn)
            self._patch_mlp(layer.mlp)
        self._attached = True
        return self

    def detach(self) -> None:
        for layer in _decoder_layers(self.model):
            for module in (layer.self_attn, layer.mlp):
                if "forward" in module.__dict__:
                    del module.__dict__["forward"]
        self._attached = False
        self.cfg = None

    def _patch_attention(self, attn) -> None:
        original = attn.forward

        def forward(hidden_states, *args, **kwargs):
            cfg = self.cfg
            if cfg is None or cfg.kv_groups == self.arch.num_kv_heads:
                return original(hidden_states, *args, **kwargs)
            return _elastic_attention(attn, self.arch, cfg, hidden_states, *args, **kwargs)

        attn.forward = forward

    def _patch_mlp(self, mlp) -> None:
        original = mlp.forward

        def forward(x):
            cfg = self.cfg
            if cfg is None or cfg.mlp_width == self.arch.intermediate_size:
                return original(x)
            return _elastic_mlp(mlp, cfg, x)

        mlp.forward = forward

    def set(self, cfg: Optional[ElasticConfig]) -> None:
        if cfg is not None:
            cfg.validate(self.arch, self.mlp_multiple)
        self.cfg = cfg

    @contextmanager
    def use(self, cfg: Optional[ElasticConfig]) -> Iterator[None]:
        """Temporarily run at ``cfg`` (``None`` for the full model)."""
        previous = self.cfg
        self.set(cfg)
        try:
            yield
        finally:
            self.cfg = previous

    def active_parameters(self, cfg: Optional[ElasticConfig] = None) -> int:
        """Parameters used by ``cfg``: all parameters minus the inactive prefix tails."""
        cfg = cfg if cfg is not None else self.cfg
        total = sum(p.numel() for p in self.model.parameters())
        if cfg is None:
            return total
        arch = self.arch
        off_groups = arch.num_kv_heads - cfg.kv_groups
        off_width = arch.intermediate_size - cfg.mlp_width
        q_off = off_groups * arch.group_size * arch.head_dim
        kv_off = off_groups * arch.head_dim
        inactive = 0
        for layer in _decoder_layers(self.model):
            attn, mlp = layer.self_attn, layer.mlp
            for proj, rows in ((attn.q_proj, q_off), (attn.k_proj, kv_off), (attn.v_proj, kv_off)):
                inactive += rows * (proj.in_features + (1 if proj.bias is not None else 0))
            inactive += q_off * attn.o_proj.out_features
            for proj in (mlp.gate_proj, mlp.up_proj):
                inactive += off_width * (proj.in_features + (1 if proj.bias is not None else 0))
            inactive += off_width * mlp.down_proj.out_features
        return total - inactive


def make_elastic(model, mlp_multiple: int = 1) -> ElasticController:
    """Attach elastic wrappers to ``model`` and return the controller."""
    return ElasticController(model, mlp_multiple=mlp_multiple).attach()
