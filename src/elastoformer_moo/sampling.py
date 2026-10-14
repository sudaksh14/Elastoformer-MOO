"""Sandwich sampling of configurations."""

from __future__ import annotations

import random
from typing import Sequence

from .arch import ArchInfo
from .config import ElasticConfig


class SandwichSampler:
    """Each draw returns the largest config, the smallest config and ``n_random`` others.

    The random picks come from the configurations strictly between the two ends and are
    sampled without replacement. With two configurations only the ends are returned.
    The sampler has its own seeded generator, so draws are reproducible and can be
    saved and restored. With ``sandwich=False`` (the naive recipe) a draw is ``n_random`` configurations
    sampled without replacement from all of them, with no guaranteed ends.
    """

    def __init__(self, configs: Sequence[ElasticConfig], arch: ArchInfo, n_random: int = 1, seed: int = 0, sandwich: bool = True):
        if not configs:
            raise ValueError("at least one configuration is required")
        if n_random < 0:
            raise ValueError("n_random must be non-negative")
        self.ordered = sorted(configs, key=lambda c: c.size(arch))
        self.n_random = n_random
        self.sandwich = sandwich
        self._rng = random.Random(seed)

    @property
    def smallest(self) -> ElasticConfig:
        return self.ordered[0]

    @property
    def largest(self) -> ElasticConfig:
        return self.ordered[-1]

    def sample(self) -> list[ElasticConfig]:
        if len(self.ordered) == 1:
            return [self.ordered[0]]
        if not self.sandwich:
            return self._rng.sample(self.ordered, min(max(self.n_random, 1), len(self.ordered)))
        picked = [self.largest, self.smallest]
        middle = self.ordered[1:-1]
        k = min(self.n_random, len(middle))
        if k:
            picked.extend(self._rng.sample(middle, k))
        return picked

    def state(self):
        return self._rng.getstate()

    def set_state(self, state) -> None:
        self._rng.setstate(state)
