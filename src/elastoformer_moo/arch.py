"""Architecture description of a decoder-only model, read from a Hugging Face config."""

from __future__ import annotations

from dataclasses import dataclass

# Model types whose decoder blocks use separate q/k/v/o and gate/up/down projections.
# This covers Llama 3.x, TinyLlama, MobileLLaMA, SmolLM/SmolLM2, OpenLLaMA (llama),
# Mistral (mistral), Qwen2/2.5 (qwen2) and Qwen3 (qwen3).
SUPPORTED_MODEL_TYPES: tuple[str, ...] = ("llama", "mistral", "qwen2", "qwen3")

_UNSUPPORTED_HINTS: dict[str, str] = {
    "phi3": "fused qkv_proj and gate_up_proj tensors need a fused-slicing adapter",
    "phi": "non-gated MLP and partial rotary embeddings need a dedicated adapter",
    "gemma": "needs an adapter for its normalization and head layout",
    "gemma2": "needs an adapter for soft-capping and alternating sliding windows",
    "gemma3": "needs an adapter for sliding-window layers and its normalization layout",
    "smollm3": "its attention layers mix rope and no-rope layers; needs an adapter",
}


class UnsupportedArchitectureError(ValueError):
    """Raised when a model type has no elastic adapter."""


@dataclass(frozen=True)
class ArchInfo:
    """Shape facts needed to define nested configurations."""

    model_type: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    tie_word_embeddings: bool

    @property
    def group_size(self) -> int:
        """Query heads served by each key-value head."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_hf_config(cls, config) -> "ArchInfo":
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            hint = _UNSUPPORTED_HINTS.get(model_type, "no adapter has been written for it")
            raise UnsupportedArchitectureError(
                f"model_type={model_type!r} is not supported ({hint}). "
                f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}."
            )
        if getattr(config, "pretraining_tp", 1) != 1:
            raise UnsupportedArchitectureError(
                "pretraining_tp != 1 uses a split-matmul path that the elastic adapter does not implement"
            )
        heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
        if heads % kv_heads:
            raise ValueError(f"num_attention_heads={heads} is not a multiple of num_key_value_heads={kv_heads}")
        head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // heads)
        return cls(
            model_type=model_type,
            num_layers=int(config.num_hidden_layers),
            hidden_size=int(config.hidden_size),
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=int(config.intermediate_size),
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )
