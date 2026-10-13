"""E1 and E2 for one model: per-configuration quality with no training, and the permutation check.

    python scripts/run_e1_e2.py --model smollm2-360m --data-dir $SCRATCH_ROOT/data/smollm2-360m \
        --out $SCRATCH_ROOT/runs/e1e2_smollm2-360m --attn flash_attention_2 --device cuda

The attention backend is explicit and has no silent fallback: flash_attention_2 fails loudly if it is
unavailable. Pass --attn sdpa to run without it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.e1e2 import format_summary, run_e1_e2  # noqa: E402
from elastoformer_moo.pipeline import ATTN_IMPLEMENTATIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="registry name, Hugging Face id, or local path")
    parser.add_argument("--data-dir", required=True, help="directory with calib.bin and the evaluation token files")
    parser.add_argument("--eval", nargs="*", default=None, metavar="NAME=FILE",
                        help="evaluation sets (default: wikitext2_test.bin and c4_val.bin in --data-dir)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--attn", default="flash_attention_2", choices=ATTN_IMPLEMENTATIONS)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=16, help="calibration blocks = batches x batch size")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mlp-multiple", type=int, default=256)
    args = parser.parse_args()

    data = Path(args.data_dir)
    if args.eval:
        eval_sets = dict(item.split("=", 1) for item in args.eval)
    else:
        eval_sets = {"wikitext2_test": data / "wikitext2_test.bin", "c4_val": data / "c4_val.bin"}
    for name, path in {**eval_sets, "calib": data / "calib.bin"}.items():
        if not Path(path).exists():
            print(f"missing token file for {name}: {path} (run scripts/prepare_eval_data.py)", file=sys.stderr)
            return 2

    results = run_e1_e2(
        args.model, args.out, data / "calib.bin", eval_sets, seq_len=args.seq_len, batch_size=args.batch_size,
        calib_batches=args.calib_batches, fractions=args.fractions, mlp_multiple=args.mlp_multiple, dtype=args.dtype,
        attn_implementation=args.attn, device=args.device,
    )
    print(format_summary(results))
    print(json.dumps(results["permutation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
