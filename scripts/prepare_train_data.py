"""Prepare the training corpus for nested training: a tokenized slice of SmolLM2's public pretraining data.

Reads parquet shards of `HuggingFaceTB/smollm-corpus` (subset `fineweb-edu-dedup`, ODC-By, ungated) by
streaming, so only the part that is needed is transferred, and tokenizes in batches. Writes two uint32 token
files into --out-dir:

  train.bin          --train-tokens tokens from shard --train-shard (default shard 0)
  fineweb_val.bin    --val-tokens tokens from a different shard (default shard 1): in-distribution held-out data

This is the same distribution the base model was pretrained on, so continued training does not shift the
data distribution. Evaluation on WikiText-2 and C4 (scripts/prepare_eval_data.py) is disjoint from it.

    python scripts/prepare_train_data.py --tokenizer smollm2-360m --out-dir $SCRATCH_ROOT/data/smollm2-360m \
        --train-tokens 520000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.data import write_token_file  # noqa: E402
from elastoformer_moo.pipeline import resolve_model_path  # noqa: E402

REPO = "HuggingFaceTB/smollm-corpus"
SUBSET = "fineweb-edu-dedup"
N_SHARDS = 234


def shard_texts(shard: int, token: str | None, max_docs: int, batch_rows: int = 2000):
    """Yield documents from one parquet shard, streaming row groups."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required (run scripts/setup_pylibs.sh)") from exc
    from huggingface_hub import HfFileSystem

    if not 0 <= shard < N_SHARDS:
        raise SystemExit(f"shard must be in [0, {N_SHARDS})")
    fs = HfFileSystem(token=token)
    path = f"datasets/{REPO}/{SUBSET}/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    produced = 0
    with fs.open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=batch_rows, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text
                produced += 1
                if produced >= max_docs:
                    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="registry name, Hugging Face id, or local path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-tokens", type=int, default=520_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--train-shard", type=int, default=0)
    parser.add_argument("--val-shard", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=20_000_000, help="safety cap on documents read")
    parser.add_argument("--batch-docs", type=int, default=512, help="documents tokenized per call")
    args = parser.parse_args()
    if args.train_shard == args.val_shard:
        raise SystemExit("train and validation must come from different shards")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # batched tokenization uses all cores
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(args.tokenizer))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"tokenizer": args.tokenizer, "vocab_size": len(tokenizer), "repo": REPO, "subset": SUBSET, "files": {}}
    for name, shard, budget in (("train.bin", args.train_shard, args.train_tokens),
                                ("fineweb_val.bin", args.val_shard, args.val_tokens)):
        start = time.time()
        written = write_token_file(shard_texts(shard, token, args.max_docs), tokenizer, out / name,
                                   max_tokens=budget, batch_size=args.batch_docs)
        seconds = time.time() - start
        manifest["files"][name] = {"shard": shard, "tokens": written, "requested": budget,
                                   "seconds": round(seconds, 1), "tokens_per_second": round(written / seconds)}
        print(f"{name}: {written:,} tokens from shard {shard} in {seconds:.0f} s", flush=True)
        if written < budget:
            print(f"WARNING: the shard ended before {budget:,} tokens; raise --max-docs or use another shard", file=sys.stderr)
    (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
