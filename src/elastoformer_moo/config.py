"""Nested configurations: how many KV groups and MLP channels each operating point keeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .arch import ArchInfo


@dataclass(frozen=True)
class ElasticConfig:
    """One operating point.

    ``kv_groups`` key-value heads (with their query heads) and ``mlp_width`` MLP
    intermediate channels are kept in every layer. Both are prefixes of the
    (importance-ordered) weights, so every smaller config is contained in every
    larger one.
    """

    name: str
    kv_groups: int
    mlp_width: int

    def validate(self, arch: ArchInfo, mlp_multiple: int = 1) -> None:
        if not 1 <= self.kv_groups <= arch.num_kv_heads:
            raise ValueError(
                f"config {self.name!r}: kv_groups={self.kv_groups} must be in [1, {arch.num_kv_heads}]"
            )
        if not 1 <= self.mlp_width <= arch.intermediate_size:
            raise ValueError(
                f"config {self.name!r}: mlp_width={self.mlp_width} must be in [1, {arch.intermediate_size}]"
            )
        if self.mlp_width != arch.intermediate_size and self.mlp_width % mlp_multiple:
            raise ValueError(
                f"config {self.name!r}: mlp_width={self.mlp_width} is not a multiple of {mlp_multiple}"
            )

    def is_full(self, arch: ArchInfo) -> bool:
        return self.kv_groups == arch.num_kv_heads and self.mlp_width == arch.intermediate_size

    def elastic_params_per_layer(self, arch: ArchInfo) -> int:
        """Weights (no biases) in the elastic projections of one layer."""
        h, hd, r, g = arch.hidden_size, arch.head_dim, arch.group_size, self.kv_groups
        attention = g * hd * h * (2 * r + 2)  # q, o: r*hd per group; k, v: hd per group
        mlp = 3 * h * self.mlp_width
        return attention + mlp

    def size(self, arch: ArchInfo) -> int:
        """Elastic parameters across all layers; used to order configurations."""
        return arch.num_layers * self.elastic_params_per_layer(arch)


def _round_to_multiple(value: float, multiple: int, upper: int) -> int:
    rounded = int(math.floor(value / multiple + 0.5)) * multiple
    return max(multiple, min(rounded, upper))


def make_grid(
    arch: ArchInfo,
    fractions: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    mlp_multiple: int = 256,
    min_kv_groups: int = 1,
) -> list[ElasticConfig]:
    """Build a grid of configurations from keep-fractions, sorted from smallest to largest.

    The fraction ``1.0`` always maps to the full model. Duplicate operating points
    (possible when the model has few KV groups) are removed.
    """
    configs: list[ElasticConfig] = []
    seen: set[tuple[int, int]] = set()
    for frac in sorted(set(fractions)):
        if not 0 < frac <= 1:
            raise ValueError(f"fraction {frac} must be in (0, 1]")
        if frac == 1:
            groups, width = arch.num_kv_heads, arch.intermediate_size
        else:
            groups = max(min_kv_groups, int(math.floor(frac * arch.num_kv_heads + 0.5)))
            width = _round_to_multiple(frac * arch.intermediate_size, mlp_multiple, arch.intermediate_size)
        if (groups, width) in seen:
            continue
        seen.add((groups, width))
        cfg = ElasticConfig(name=f"f{int(round(frac * 100)):03d}", kv_groups=groups, mlp_width=width)
        cfg.validate(arch, mlp_multiple)
        configs.append(cfg)
    return sorted(configs, key=lambda c: c.size(arch))


def smallest_and_largest(configs: Sequence[ElasticConfig], arch: ArchInfo) -> tuple[ElasticConfig, ElasticConfig]:
    ordered = sorted(configs, key=lambda c: c.size(arch))
    return ordered[0], ordered[-1]
