"""Print the nodes to exclude so a Slurm job can only land on allowed nodes.

    python scripts/node_policy.py exclude gpu "node201 node202 ..."
    python scripts/node_policy.py check gpu node205

Loads cluster.py by path so it works with a bare system Python (no torch required).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "elastoformer_cluster", Path(__file__).resolve().parents[1] / "src" / "elastoformer_moo" / "cluster.py"
)
cluster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cluster)


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in ("exclude", "check"):
        print(__doc__, file=sys.stderr)
        return 2
    command, kind, value = argv[0], argv[1], " ".join(argv[2:])
    if command == "check":
        return 0 if cluster.is_allowed(value, kind) else 3
    print(",".join(cluster.exclude_list(value.split(), kind)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(3)
