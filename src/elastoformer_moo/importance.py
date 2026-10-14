"""Calibration-based importance of MLP channels and KV groups.

Scores (higher is more important):

* MLP channel ``j``: mean over tokens of ``|h_j|`` times the L2 norm of column ``j`` of
  ``down_proj``, where ``h`` is the activated gate times up output. This is a
  Wanda-style activation-times-weight heuristic, not a claim of optimality.
* KV group ``g``: sum over its query heads of the mean token norm of the head output
  times the Frobenius norm of the matching ``o_proj`` columns.

``method="taylor"`` replaces both with a first-order estimate of the loss change from removing the
structure: the mean over tokens of ``|x * dL/dx|``, with ``x`` the input of ``down_proj`` (per channel)
or of ``o_proj`` (per head, summed over ``head_dim``, then over the heads of a KV group). It needs a
backward pass but no parameter gradients. ``skip_first_token`` leaves position 0 out of the statistics:
in models with an attention sink that token carries massive activations that otherwise dominate the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .arch import ArchInfo
from .losses import lm_cross_entropy
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
    method: str = "act_weight",
    skip_first_token: bool = False,
) -> Importance:
    """Run ``batches`` (``[B, T]`` token ids) through the full model and score structures."""
    if method not in ("act_weight", "taylor"):
        raise ValueError(f"method must be 'act_weight' or 'taylor', got {method!r}")
    taylor = method == "taylor"
    first = 1 if skip_first_token else 0
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    mlp_sum = [torch.zeros(arch.intermediate_size, device=device, dtype=torch.float32) for _ in layers]
    head_sum = [torch.zeros(arch.num_heads, device=device, dtype=torch.float32) for _ in layers]
    tokens = 0

    handles = []
    for idx, layer in enumerate(layers):

        def mlp_hook(_mod, args, idx=idx):
            x = args[0]
            if taylor:
                def acc(g, x=x, idx=idx):
                    mlp_sum[idx].add_((x.detach() * g)[:, first:].abs().float().sum(dim=(0, 1)))

                x.register_hook(acc)
            else:
                mlp_sum[idx] += x.detach()[:, first:].abs().float().sum(dim=(0, 1))

        def attn_hook(_mod, args, idx=idx):
            x = args[0]
            b, t, _ = x.shape
            if taylor:
                def acc(g, x=x, idx=idx):
                    prod = (x.detach() * g).float().view(b, t, arch.num_heads, arch.head_dim).sum(dim=-1)
                    head_sum[idx].add_(prod[:, first:].abs().sum(dim=(0, 1)))

                x.register_hook(acc)
            else:
                ctx = x.detach().float()
                head_sum[idx] += ctx.view(b, t, arch.num_heads, arch.head_dim).norm(dim=-1)[:, first:].sum(dim=(0, 1))

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
                if taylor:
                    # gradients w.r.t. activations only: a leaf embedding input, parameters stay untouched
                    with torch.enable_grad():
                        emb = model.get_input_embeddings()(batch).detach().requires_grad_(True)
                        logits = model(inputs_embeds=emb, use_cache=False).logits
                        lm_cross_entropy(logits, batch).backward(inputs=[emb])
                else:
                    model(input_ids=batch)
                tokens += batch.shape[0] * (batch.shape[1] - first)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if tokens == 0:
        raise ValueError("no calibration tokens were processed")

    mlp_scores, group_scores = [], []
    for idx, layer in enumerate(layers):
        down_norm = layer.mlp.down_proj.weight.detach().float().norm(dim=0)  # per intermediate channel
        mlp_scores.append((mlp_sum[idx] / tokens * (1 if taylor else down_norm)).cpu())

        o_weight = layer.self_attn.o_proj.weight.detach().float()
        o_norm = o_weight.view(o_weight.shape[0], arch.num_heads, arch.head_dim).norm(dim=(0, 2))
        head_scores = head_sum[idx] / tokens * (1 if taylor else o_norm)
        group_scores.append(head_scores.view(arch.num_kv_heads, arch.group_size).sum(dim=1).cpu())
    return Importance(mlp=mlp_scores, groups=group_scores)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
