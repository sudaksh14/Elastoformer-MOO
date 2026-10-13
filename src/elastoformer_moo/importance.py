"""Calibration-based importance of MLP channels and KV groups.

Scores (higher is more important):

* MLP channel ``j``: mean over tokens of ``|h_j|`` times the L2 norm of column ``j`` of
  ``down_proj``, where ``h`` is the activated gate times up output. This is a
  Wanda-style activation-times-weight heuristic, not a claim of optimality.
* KV group ``g``: sum over its query heads of the mean token norm of the head output
  times the Frobenius norm of the matching ``o_proj`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .arch import ArchInfo
from .nested import ElasticController, _decoder_layers


@dataclass
class Importance:
    mlp: list[torch.Tensor]  # per layer, shape [intermediate_size]
    groups: list[torch.Tensor]  # per layer, shape [num_kv_heads]

    def save(self, path: str) -> None:
        torch.save({"mlp": self.mlp, "groups": self.groups}, path)

    @classmethod
    def load(cls, path: str) -> "Importance":
        blob = torch.load(path, map_location="cpu")
        return cls(mlp=blob["mlp"], groups=blob["groups"])


@torch.no_grad()
def collect_importance(
    model,
    arch: ArchInfo,
    batches: Iterable[torch.Tensor],
    controller: Optional[ElasticController] = None,
    max_batches: Optional[int] = None,
) -> Importance:
    """Run ``batches`` (``[B, T]`` token ids) through the full model and score structures."""
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    mlp_sum = [torch.zeros(arch.intermediate_size, device=device, dtype=torch.float32) for _ in layers]
    head_sum = [torch.zeros(arch.num_heads, device=device, dtype=torch.float32) for _ in layers]
    tokens = 0

    handles = []
    for idx, layer in enumerate(layers):

        def mlp_hook(_mod, args, idx=idx):
            mlp_sum[idx] += args[0].detach().abs().float().sum(dim=(0, 1))

        def attn_hook(_mod, args, idx=idx):
            ctx = args[0].detach().float()
            b, t, _ = ctx.shape
            head_sum[idx] += ctx.view(b, t, arch.num_heads, arch.head_dim).norm(dim=-1).sum(dim=(0, 1))

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_hook))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_hook))

    was_training = model.training
    model.eval()
    try:
        context = controller.use(None) if controller is not None else _null_context()
        with context:
            for count, batch in enumerate(batches):
                if max_batches is not None and count >= max_batches:
                    break
                batch = batch.to(device)
                model(input_ids=batch)
                tokens += batch.numel()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if tokens == 0:
        raise ValueError("no calibration tokens were processed")

    mlp_scores, group_scores = [], []
    for idx, layer in enumerate(layers):
        down_norm = layer.mlp.down_proj.weight.detach().float().norm(dim=0)  # per intermediate channel
        mlp_scores.append((mlp_sum[idx] / tokens * down_norm).cpu())

        o_weight = layer.self_attn.o_proj.weight.detach().float()
        o_norm = o_weight.view(o_weight.shape[0], arch.num_heads, arch.head_dim).norm(dim=(0, 2))
        head_scores = head_sum[idx] / tokens * o_norm
        group_scores.append(head_scores.view(arch.num_kv_heads, arch.group_size).sum(dim=1).cpu())
    return Importance(mlp=mlp_scores, groups=group_scores)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
