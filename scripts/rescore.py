"""Re-score a checkpoint on the full evaluation files, with one protocol for every run.

The training-time evaluation read only the first ``eval_batches`` batches of each file. The FineWeb-Edu file has
about 4,900 blocks and the training evaluation used 320 of them, so this script reads every block of every file.
For a hypervolume-weighted run the last ``hv_val_batches * batch_size`` blocks of the FineWeb-Edu file are excluded,
because the weights were selected on them.

    python scripts/rescore.py --model $SCRATCH/runs/<run>/final --run-dir $SCRATCH/runs/<run> \
        --data-dir $SCRATCH/data/smollm2-360m --out $SCRATCH/runs/<run>/eval_full.json

Use --model $SCRATCH/models/smollm2-360m-permuted (and no --run-dir) for the untrained starting point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.data import TokenBlocks  # noqa: E402
from elastoformer_moo.evaluate import evaluate_configs  # noqa: E402
from elastoformer_moo.nested import make_elastic  # noqa: E402
from elastoformer_moo.pipeline import load_causal_lm  # noqa: E402

SETS = {"wikitext2": "wikitext2_test.bin", "c4": "c4_val.bin", "fineweb": "fineweb_val.bin"}


def blocks_in_order(blocks: TokenBlocks, batch_size: int, n_blocks: int):
    for first in range(0, n_blocks, batch_size):
        yield torch.stack([blocks.block(i) for i in range(first, min(first + batch_size, n_blocks))])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint directory (a run's final/ or the permuted model)")
    parser.add_argument("--run-dir", default=None, help="run directory, to read train_config.json (HV exclusion)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mlp-multiple", type=int, default=256)
    parser.add_argument("--attn", default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    excluded_name, excluded_blocks = None, 0
    if args.run_dir:
        tcfg = json.loads((Path(args.run_dir) / "train_config.json").read_text())
        if tcfg.get("weight_mode") == "hv":
            excluded_name = tcfg.get("hv_val_set", "fineweb")
            excluded_blocks = tcfg.get("hv_val_batches", 2) * args.batch_size

    # same numerics as training: fp32 master weights, bf16 autocast, the attention backend asked for explicitly
    lm = load_causal_lm(args.model, "float32", args.device, attn_implementation=args.attn, autocast=True)
    ctrl = make_elastic(lm, mlp_multiple=args.mlp_multiple)
    configs = make_grid(ctrl.arch, args.fractions, mlp_multiple=args.mlp_multiple)

    report = {"model": args.model, "run_dir": args.run_dir, "protocol": {
        "seq_len": args.seq_len, "batch_size": args.batch_size, "attn": args.attn, "autocast_bf16": True,
        "all_blocks": True, "excluded": {"set": excluded_name, "blocks": excluded_blocks}}, "sets": {}}
    for name, filename in SETS.items():
        blocks = TokenBlocks(Path(args.data_dir) / filename, args.seq_len)
        n_blocks = len(blocks) - (excluded_blocks if name == excluded_name else 0)
        rows = evaluate_configs(
            lm, ctrl, configs, lambda b=blocks, n=n_blocks: blocks_in_order(b, args.batch_size, n), autocast_bf16=True)
        report["sets"][name] = {"blocks": n_blocks, "tokens": rows[0]["tokens"], "rows": rows}
        for row in rows:
            print(name, row["config"], round(row["perplexity"], 3), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
