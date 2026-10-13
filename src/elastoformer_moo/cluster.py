"""Node placement policy for DAS-6 jobs.

GPU work must run on nodes 205-208 and CPU-only work on nodes 201-204. The default patterns
assume node names like ``node205``; that naming is not verified against the cluster. Override
with ``ELASTOFORMER_ALLOWED_GPU_NODES`` / ``ELASTOFORMER_ALLOWED_CPU_NODES`` (regular
expressions matched against the whole node name) once the real names are known.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

DEFAULT_PATTERNS = {"gpu": r"node20[5-8]", "cpu": r"node20[1-4]"}
_ENV = {"gpu": "ELASTOFORMER_ALLOWED_GPU_NODES", "cpu": "ELASTOFORMER_ALLOWED_CPU_NODES"}


def pattern(kind: str) -> str:
    if kind not in DEFAULT_PATTERNS:
        raise ValueError(f"kind must be 'gpu' or 'cpu', got {kind!r}")
    return os.environ.get(_ENV[kind], DEFAULT_PATTERNS[kind])


def is_allowed(node: str, kind: str) -> bool:
    return re.fullmatch(pattern(kind), node.strip()) is not None


def exclude_list(all_nodes: Iterable[str], kind: str) -> list[str]:
    """Nodes to pass to ``sbatch --exclude`` so the job can only land on allowed nodes.

    Slurm's ``--nodelist`` forces *all* listed nodes into the job, so it cannot express "any
    one of these". Excluding the complement can.
    """
    nodes = [n.strip() for n in all_nodes if n.strip()]
    allowed = [n for n in nodes if is_allowed(n, kind)]
    if not allowed:
        raise ValueError(
            f"none of the {len(nodes)} cluster nodes matches the {kind} pattern {pattern(kind)!r}; "
            "refusing to build an exclude list that would leave no valid node"
        )
    return sorted(set(nodes) - set(allowed))
