"""Per-configuration evaluation."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch

from .config import ElasticConfig
from .losses import lm_cross_entropy
from .nested import ElasticController


@torch.no_grad()
def perplexity(
    model,
    controller: ElasticController,
    cfg: Optional[ElasticConfig],
    batches: Iterable[torch.Tensor],
    autocast_bf16: bool = False,
) -> dict:
    """Token-weighted perplexity of ``cfg`` (``None`` for the full model) over ``batches``.

    ``autocast_bf16`` runs the forward pass under bf16 autocast. Needed for float32 weights with flash
    attention, which only accepts half-precision inputs (the same setting training uses).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with controller.use(cfg), torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast_bf16):
        for batch in batches:
            batch = batch.to(device)
            logits = model(input_ids=batch).logits
            n_tokens = batch.size(0) * (batch.size(1) - 1)
            total_nll += lm_cross_entropy(logits, batch).item() * n_tokens
            total_tokens += n_tokens
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("no evaluation tokens were processed")
    nll = total_nll / total_tokens
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": total_tokens}


def evaluate_configs(
    model,
    controller: ElasticController,
    configs: Sequence[ElasticConfig],
    make_batches,
    autocast_bf16: bool = False,
) -> list[dict]:
    """Evaluate each configuration; ``make_batches()`` must return a fresh iterable each call."""
    rows = []
    for cfg in configs:
        row = {
            "config": cfg.name,
            "kv_groups": cfg.kv_groups,
            "mlp_width": cfg.mlp_width,
            "active_parameters": controller.active_parameters(cfg),
        }
        row.update(perplexity(model, controller, cfg, make_batches(), autocast_bf16))
        rows.append(row)
    return rows
