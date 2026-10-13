"""Joint nested training from a YAML experiment config.

    python scripts/train_nested.py --config configs/experiments/llama3.2-1b_nested.yaml \
        --set train.steps=50 --set out_dir=$SCRATCH/runs/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.pipeline import apply_overrides, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    apply_overrides(cfg, args.overrides)
    main_rank = os.environ.get("RANK", "0") == "0"  # under torchrun every process runs this; print once
    if main_rank:
        print(json.dumps(cfg, indent=2, default=str))
    result = run_training(cfg)
    if main_rank:
        print(json.dumps({k: v for k, v in result.items() if k != "eval"}, indent=2))
        evals = result.get("eval", [])
        for name, rows in (evals.items() if isinstance(evals, dict) else [("val", evals)]):
            for row in rows:
                print(name, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
