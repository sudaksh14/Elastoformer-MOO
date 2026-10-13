"""Perplexity of every nested configuration of a checkpoint (no training).

    python scripts/eval_ppl.py --model $SCRATCH/models/llama3.2-1b-permuted \
        --tokens $SCRATCH/data/val.bin --seq-len 1024 --batch-size 4 --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.pipeline import evaluate_pretrained  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mlp-multiple", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn", default=None, choices=["eager", "sdpa", "flash_attention_2"],
                        help="attention backend; flash_attention_2 needs --dtype bfloat16 or float16 and never falls back silently")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = evaluate_pretrained(
        args.model, args.tokens, args.seq_len, args.batch_size, args.fractions, args.mlp_multiple,
        args.max_batches, args.dtype, args.device, args.attn,
    )
    for row in rows:
        print(row)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
