"""Per-configuration loss weights: equal, a fixed boost for the full model, or hypervolume-driven.

Loss of one micro-batch is the weighted mean ``sum_c w_c * L_c / sum_c w_c`` over the sampled configurations,
so the gradient scale does not change with the weights.

``hv`` mode (first-order bilevel, adapted from GMOOAR in Chen 2026, Ch. 3, to nested configurations):
one inner SGD-like step on configuration ``j``'s training gradient ``g_j`` changes the validation loss of
configuration ``i`` by ``-lr * <v_i, g_j>`` with ``v_i`` the validation gradient. Weighting the objectives by
the hypervolume sensitivities ``width_i`` (cost.hv_widths), the hypervolume gain from training on ``j`` is
proportional to ``u_j = sum_i width_i <v_i, g_j>``. The outer step raises ``w_j`` where ``u_j`` is large:
``w_j <- clip(w_j * exp(beta * u_j / max|u|))`` and renormalises to mean 1. ``u`` is smoothed by an EMA.

The alignments are cosines, not raw dot products. A raw dot product scales with the gradient norm, and the
smallest configurations have the largest loss and gradients, so raw dots let them dominate every update (the
first run collapsed the full model's weight to its floor for that reason). The largest configuration's weight
never drops below ``full_min`` (default: its starting weight of 1), so the update cannot sacrifice the full model.

This is an approximation (one step lookahead, one training and a few validation batches), not an exact
implicit gradient.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import torch

from .config import ElasticConfig
from .cost import CostFn, hv_widths

MODES = ("fixed", "hv")


class ConfigWeights:
    def __init__(self, configs: Sequence[ElasticConfig], arch, mode: str = "fixed", max_weight: float = 1.0,
                 w_min: float = 0.25, w_max: float = 4.0, beta: float = 0.5, ema: float = 0.9, full_min: float = 1.0):
        if mode not in MODES:
            raise ValueError(f"weight_mode must be one of {MODES}, got {mode!r}")
        if max_weight <= 0 or w_min <= 0 or w_max < w_min:
            raise ValueError("weights must be positive and w_max >= w_min")
        self.mode, self.w_min, self.w_max, self.beta, self.ema = mode, w_min, w_max, beta, ema
        self.full_min = full_min
        self.names = [c.name for c in sorted(configs, key=lambda c: c.size(arch))]
        self.w = {n: 1.0 for n in self.names}
        self.w[self.names[-1]] = float(max_weight)  # the largest configuration (fixed mode; hv starts from the same point)
        self._u: Optional[list[float]] = None

    def as_dict(self) -> dict[str, float]:
        return dict(self.w)

    def state(self) -> dict:
        return {"w": dict(self.w), "u": self._u}

    def set_state(self, state: Mapping) -> None:
        self.w, self._u = dict(state["w"]), state["u"]

    def update(self, utilities: Mapping[str, float]) -> dict[str, float]:
        """Outer step: multiplicative update from per-configuration utilities ``u_j``."""
        u = [float(utilities[n]) for n in self.names]
        self._u = u if self._u is None else [self.ema * a + (1 - self.ema) * b for a, b in zip(self._u, u)]
        scale = max(abs(x) for x in self._u) or 1.0
        new = {n: self.w[n] * math.exp(self.beta * x / scale) for n, x in zip(self.names, self._u)}
        mean = sum(new.values()) / len(new)
        self.w = {n: min(self.w_max, max(self.w_min, v / mean)) for n, v in new.items()}
        largest = self.names[-1]
        self.w[largest] = max(self.w[largest], self.full_min)
        return self.as_dict()


def hv_utilities(grad_dots: Sequence[Sequence[float]], costs: Sequence[float], val_losses: Sequence[float],
                 ref_scale: float = 1.25) -> list[float]:
    """``u_j = sum_i width_i * dots[i][j]`` with ``dots[i][j]`` the cosine between the validation gradient of ``i``
    and the training gradient of ``j``."""
    widths = hv_widths(costs, val_losses, ref_scale)
    n = len(widths)
    return [sum(widths[i] * grad_dots[i][j] for i in range(n)) for j in range(n)]


def gradient_dots(model, controller, configs: Sequence[ElasticConfig], train_batch: torch.Tensor,
                  val_batches: Sequence[torch.Tensor], loss_fn, autocast_bf16: bool):
    """Return (cos, val_losses): ``cos[i][j]`` is the cosine between the gradient of validation loss ``i`` and the
    gradient of training loss ``j``, over all trainable parameters.

    ``loss_fn(model, controller, cfg, batch, training) -> scalar``. Uses ``torch.autograd.grad`` so ``.grad``
    of the parameters (and the optimizer's accumulation) is untouched.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n = len(configs)
    def norm(grads) -> float:
        return math.sqrt(sum(float((g * g).sum()) for g in grads if g is not None))

    train_grads, train_norms = [], []
    for cfg in configs:
        loss = loss_fn(model, controller, cfg, train_batch, True)
        grads = [g.detach() for g in torch.autograd.grad(loss, params, allow_unused=True)]
        train_grads.append(grads)
        train_norms.append(norm(grads))
    dots, val_losses = [], []
    for cfg in configs:
        total, row = 0.0, [0.0] * n
        grads_acc = None
        for batch in val_batches:
            loss = loss_fn(model, controller, cfg, batch, False)
            total += float(loss.detach())
            grads = [None if g is None else g.detach() for g in torch.autograd.grad(loss, params, allow_unused=True)]
            grads_acc = grads if grads_acc is None else [a if b is None else (b if a is None else a + b) for a, b in zip(grads_acc, grads)]
        scale = 1.0 / len(val_batches)  # mean validation loss; cosines are scale-free
        val_norm = norm(grads_acc)
        for j in range(n):
            dot = sum(float((a * b).sum()) for a, b in zip(grads_acc, train_grads[j]) if a is not None and b is not None)
            row[j] = dot / (val_norm * train_norms[j] + 1e-12)
        dots.append(row)
        val_losses.append(total * scale)
    return dots, val_losses
