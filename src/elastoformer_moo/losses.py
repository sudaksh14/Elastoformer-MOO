"""Losses for joint nested training."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def lm_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy: position ``t`` predicts token ``t + 1``."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), input_ids[:, 1:].reshape(-1), reduction="mean"
    )


def distillation_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Mean per-token KL(teacher || student), scaled by ``temperature ** 2``."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"logit shapes differ: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = F.log_softmax(student_logits[:, :-1].float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1].float() / temperature, dim=-1)
    kl = (t.exp() * (t - s)).sum(dim=-1)
    return kl.mean() * temperature**2


def monotonicity_penalty(losses: Sequence[torch.Tensor], sizes: Sequence[int]) -> torch.Tensor:
    """Penalty ``sum_i relu(L_larger - L_smaller)`` over configurations ordered by size.

    A larger nested configuration contains every smaller one, so it should not have a
    higher loss. Returns zero when ``losses`` has fewer than two entries.
    """
    if len(losses) != len(sizes):
        raise ValueError("losses and sizes must have the same length")
    if len(losses) < 2:
        return torch.zeros((), device=losses[0].device if losses else None)
    ordered = [loss for _, loss in sorted(zip(sizes, losses), key=lambda pair: pair[0])]
    return sum(F.relu(ordered[i + 1] - ordered[i]) for i in range(len(ordered) - 1))
