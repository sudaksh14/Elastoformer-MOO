"""Prepare the standard perplexity data for one tokenizer.

Writes three uint32 token files into --out-dir:
  wikitext2_test.bin  WikiText-2 (raw) test split, rows joined with "\\n\\n" (the GPTQ/SparseGPT/Wanda protocol)
  c4_val.bin          the first documents of the C4 validation shard en/c4-validation.00000-of-00008
  calib.bin           the first documents of the C4 training shard en/c4-train.00000-of-01024, for importance
                      scoring (the same source Wanda and SparseGPT calibrate on)

C4 shards are streamed, so only the first part of each ~40-320 MB file is transferred. WikiText-2 is read
from parquet and needs pyarrow (scripts/setup_pylibs.sh installs it into the overlay).

    python scripts/prepare_eval_data.py --tokenizer smollm2-360m --out-dir $SCRATCH_ROOT/data/smollm2-360m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.data import write_token_file  # noqa: E402
from elastoformer_moo.pipeline import resolve_model_path  # noqa: E402
from elastoformer_moo.textdata import iter_jsonl_gz_texts, wikitext_join  # noqa: E402

WIKITEXT_REPO = "Salesforce/wikitext"
C4_REPO = "allenai/c4"
C4_VAL = "en/c4-validation.00000-of-00008.json.gz"
C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"


def wikitext2_text(split: str, token: str | None) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read WikiText-2 (run scripts/setup_pylibs.sh, or pip install pyarrow)") from exc
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKITEXT_REPO, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset", token=token)
    return wikitext_join(pq.read_table(path).column("text").to_pylist())


def c4_texts(filename: str, token: str | None, max_docs: int):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{C4_REPO}/{filename}", "rb") as handle:
        yield from iter_jsonl_gz_texts(handle, max_docs=max_docs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="registry name, Hugging Face id, or local path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calib-tokens", type=int, default=131072, help="default 128 blocks of 1024")
    parser.add_argument("--c4-val-tokens", type=int, default=262144, help="default 256 blocks of 1024")
    parser.add_argument("--max-docs", type=int, default=20000, help="safety cap on documents read per C4 shard")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(args.tokenizer))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    # WikiText-2: one string, tokenized once, no EOS added between rows (matches the reference protocol)
    counts["wikitext2_test.bin"] = write_token_file([wikitext2_text("test", token)], tokenizer, out / "wikitext2_test.bin", append_eos=False)
    counts["c4_val.bin"] = write_token_file(c4_texts(C4_VAL, token, args.max_docs), tokenizer, out / "c4_val.bin", max_tokens=args.c4_val_tokens)
    counts["calib.bin"] = write_token_file(c4_texts(C4_TRAIN, token, args.max_docs), tokenizer, out / "calib.bin", max_tokens=args.calib_tokens)

    manifest = {
        "tokenizer": args.tokenizer, "vocab_size": len(tokenizer), "tokens": counts,
        "sources": {"wikitext2_test.bin": f"{WIKITEXT_REPO} wikitext-2-raw-v1 test", "c4_val.bin": f"{C4_REPO} {C4_VAL}",
                    "calib.bin": f"{C4_REPO} {C4_TRAIN}"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    short = [k for k, v in counts.items() if k != "wikitext2_test.bin" and v < (args.calib_tokens if k == "calib.bin" else args.c4_val_tokens)]
    if short:
        print(f"WARNING: fewer tokens than requested for {short}; raise --max-docs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
