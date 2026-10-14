"""Importance-order a pretrained model so every nested configuration is a prefix.

    python scripts/permute_model.py --model llama3.2-1b --calib $SCRATCH/data/calib.bin \
        --out $SCRATCH/models/llama3.2-1b-permuted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.pipeline import permute_pretrained  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registry name, Hugging Face id, or local path")
    parser.add_argument("--calib", required=True, help="token file used to score importance")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-batches", type=int, default=32)
    parser.add_argument("--compute-dtype", default="float32")
    parser.add_argument("--save-dtype", default="bfloat16")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--importance-method", default="act_weight", choices=["act_weight", "taylor"])
    parser.add_argument("--skip-first-token", action="store_true", help="leave position 0 (attention sink) out of the statistics")
    args = parser.parse_args()
    report = permute_pretrained(
        args.model, args.out, args.calib, args.seq_len, args.batch_size, args.n_batches,
        args.compute_dtype, args.save_dtype, args.atol, args.rtol, args.device,
        args.importance_method, args.skip_first_token,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
