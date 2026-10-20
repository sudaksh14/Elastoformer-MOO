"""Cost axis of a configuration: what "smaller" means when configurations are compared.

The axis is chosen by a spec string (``train.cost_axis``), so the same code can rank configurations by
FLOPs, by memory, or by anything measured on a device (latency, energy):

* ``params``    elastic parameters (the default; also what orders configurations by nesting)
* ``flops``     forward FLOPs per token at ``seq_len`` (matmuls of the elastic projections plus attention scores)
* ``kv_cache``  KV-cache bytes per token (2 bytes per element)
* ``table:PATH`` measured values from a JSON file ``{"f025": 0.012, "f050": 0.019, ...}``; the unit is yours
                 (seconds, joules, bytes) and must cover every configuration name

Lower cost is better on every axis. Sampling and the monotonicity penalty keep ordering configurations by
nesting (parameters): a measured cost need not be monotone, and "largest" must stay the full model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from .arch import ArchInfo
from .config import ElasticConfig

CostFn = Callable[[ElasticConfig], float]
BUILTIN_AXES = ("params", "flops", "kv_cache")


def make_cost_fn(spec: str, arch: ArchInfo, configs: Sequence[ElasticConfig], seq_len: int = 1024) -> CostFn:
    """Return ``cfg -> cost`` for ``spec``; raise ``ValueError`` for an unknown spec or an incomplete table."""
    if spec == "params":
        return lambda c: float(c.size(arch))
    if spec == "flops":
        # 2 FLOPs per multiply-add of every elastic weight, plus QK^T and AV: 4 * seq * head_dim per query head
        return lambda c: float(
            arch.num_layers * (2 * c.elastic_params_per_layer(arch) + 4 * seq_len * arch.head_dim * c.kv_groups * arch.group_size)
        )
    if spec == "kv_cache":
        return lambda c: float(2 * arch.num_layers * c.kv_groups * arch.head_dim * 2)
    if spec.startswith("table:"):
        path = Path(spec[len("table:"):])
        table = json.loads(path.read_text(encoding="utf-8"))
        missing = [c.name for c in configs if c.name not in table]
        if missing:
            raise ValueError(f"cost table {path} has no entry for configuration(s) {missing}")
        values = {name: float(v) for name, v in table.items()}
        return lambda c: values[c.name]
    raise ValueError(f"unknown cost axis {spec!r}; use one of {BUILTIN_AXES} or 'table:<path.json>'")


def hypervolume_2d(points: Sequence[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Area dominated by ``points`` (cost, loss), both minimised, bounded by ``reference``.

    Points that do not strictly improve on the reference in both coordinates add nothing.
    """
    ref_cost, ref_loss = reference
    useful = sorted((c, l) for c, l in points if c < ref_cost and l < ref_loss)
    area, best_loss = 0.0, ref_loss
    for i, (cost, loss) in enumerate(useful):
        best_loss = min(best_loss, loss)
        next_cost = useful[i + 1][0] if i + 1 < len(useful) else ref_cost
        area += (next_cost - cost) * (ref_loss - best_loss)
    return area


def hv_widths(costs: Sequence[float], losses: Sequence[float], ref_scale: float = 1.25) -> list[float]:
    """Sensitivity of the hypervolume to each configuration's loss: ``d HV / d(-loss_i)``.

    The hypervolume of points ``(cost_i, loss_i)`` (both minimised) against a reference whose cost is
    ``ref_scale * max(cost)`` is a staircase. Lowering the loss of a non-dominated point by ``d`` adds
    ``d * width`` of area, where ``width`` is the cost distance to the next non-dominated point (or to the
    reference for the last one). A dominated point (a configuration that is worse than a cheaper one) has
    sensitivity 0. The reference loss does not enter the derivative.
    """
    if len(costs) != len(losses) or not costs:
        raise ValueError("costs and losses must be non-empty and of equal length")
    if ref_scale <= 1.0:
        raise ValueError("ref_scale must exceed 1 so the most expensive point lies inside the reference")
    order = sorted(range(len(costs)), key=lambda i: (costs[i], losses[i]))
    nondominated, best = [], float("inf")
    for i in order:
        if losses[i] < best:
            nondominated.append(i)
            best = losses[i]
    reference = ref_scale * max(costs)
    widths = [0.0] * len(costs)
    for k, i in enumerate(nondominated):
        upper = costs[nondominated[k + 1]] if k + 1 < len(nondominated) else reference
        widths[i] = upper - costs[i]
    return widths
