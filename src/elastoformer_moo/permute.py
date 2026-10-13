"""Reorder weights once so that every configuration is a prefix of the most important structures.

MLP channels are permuted jointly in gate, up (rows) and down (columns). KV groups are
permuted jointly across q (rows of the group's query heads), k and v (rows of the
group's KV head) and o (the matching columns). Norm weights act on head_dim or the
residual stream, so they are not touched. The permuted model computes the same
function; :func:`verify_permutation_equivalence` checks that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .arch import ArchInfo
from .importance import Importance
from .nested import _decoder_layers


@dataclass
class Permutation:
    mlp: list[torch.Tensor]  # per layer, new position -> old channel
    groups: list[torch.Tensor]  # per layer, new position -> old group

    def save(self, path: str) -> None:
        torch.save({"mlp": self.mlp, "groups": self.groups}, path)

    @classmethod
    def load(cls, path: str) -> "Permutation":
        blob = torch.load(path, map_location="cpu")
        return cls(mlp=blob["mlp"], groups=blob["groups"])


def permutation_from_importance(importance: Importance) -> Permutation:
    """Most important first; ties keep their original order."""

    def order(score: torch.Tensor) -> torch.Tensor:
        return torch.argsort(score, descending=True, stable=True)

    return Permutation(mlp=[order(s) for s in importance.mlp], groups=[order(s) for s in importance.groups])


def _group_index(order: torch.Tensor, rows_per_group: int) -> torch.Tensor:
    return torch.cat([torch.arange(g * rows_per_group, (g + 1) * rows_per_group) for g in order.tolist()])


@torch.no_grad()
def apply_permutation(model, arch: ArchInfo, perm: Permutation) -> None:
    """Permute ``model`` in place."""
    layers = _decoder_layers(model)
    if len(perm.mlp) != len(layers) or len(perm.groups) != len(layers):
        raise ValueError("permutation has a different number of layers than the model")
    for layer, mlp_order, group_order in zip(layers, perm.mlp, perm.groups):
        if mlp_order.numel() != arch.intermediate_size or group_order.numel() != arch.num_kv_heads:
            raise ValueError("permutation size does not match the architecture")
        mlp, attn = layer.mlp, layer.self_attn

        idx = mlp_order.to(mlp.gate_proj.weight.device)
        for proj in (mlp.gate_proj, mlp.up_proj):
            proj.weight.data = proj.weight.data[idx].clone()
            if proj.bias is not None:
                proj.bias.data = proj.bias.data[idx].clone()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, idx].clone()

        q_idx = _group_index(group_order, arch.group_size * arch.head_dim).to(attn.q_proj.weight.device)
        kv_idx = _group_index(group_order, arch.head_dim).to(attn.q_proj.weight.device)
        attn.q_proj.weight.data = attn.q_proj.weight.data[q_idx].clone()
        attn.k_proj.weight.data = attn.k_proj.weight.data[kv_idx].clone()
        attn.v_proj.weight.data = attn.v_proj.weight.data[kv_idx].clone()
        for proj, index in ((attn.q_proj, q_idx), (attn.k_proj, kv_idx), (attn.v_proj, kv_idx)):
            if proj.bias is not None:
                proj.bias.data = proj.bias.data[index].clone()
        attn.o_proj.weight.data = attn.o_proj.weight.data[:, q_idx].clone()


@torch.no_grad()
def verify_permutation_equivalence(
    model, arch: ArchInfo, perm: Permutation, batch: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4
) -> float:
    """Apply ``perm`` and check the logits are unchanged. Returns the maximum absolute difference.

    The check passes when ``max|diff| <= atol + rtol * max|logit|``. The relative term is
    needed because a permutation reorders the summation in every matmul: on a real 30-layer
    float32 model whose logits reach about 40 this alone changes them by ~2e-4, while a real
    indexing error changes them by O(1) or more.

    The model is left permuted when the check passes. If it fails, the permutation is
    undone and ``AssertionError`` is raised.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    before = model(input_ids=batch.to(device)).logits.float()
    apply_permutation(model, arch, perm)
    after = model(input_ids=batch.to(device)).logits.float()
    diff = (before - after).abs().max().item()
    tolerance = atol + rtol * before.abs().max().item()
    model.train(was_training)
    if diff > tolerance:
        apply_permutation(model, arch, _inverse(perm))
        raise AssertionError(
            f"permutation changed the logits by {diff:.3e} (tolerance {tolerance:.3e} = "
            f"atol {atol:.1e} + rtol {rtol:.1e} * max|logit|)"
        )
    return diff


def _inverse(perm: Permutation) -> Permutation:
    def inv(order: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(order)
        out[order] = torch.arange(order.numel())
        return out

    return Permutation(mlp=[inv(o) for o in perm.mlp], groups=[inv(o) for o in perm.groups])
