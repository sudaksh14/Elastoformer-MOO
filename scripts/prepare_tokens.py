"""Tokenize text into a flat uint32 token file for training and evaluation.

Input is plain-text files (one document per file) or JSONL files with a text field.

    python scripts/prepare_tokens.py --tokenizer meta-llama/Llama-3.2-1B \
        --inputs corpus/*.txt --out $SCRATCH/data/train.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.data import write_token_file  # noqa: E402
from elastoformer_moo.pipeline import resolve_model_path  # noqa: E402


def iter_texts(paths, field):
    for path in paths:
        path = Path(path)
        if path.suffix in (".jsonl", ".json"):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)[field]
        else:
            yield path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="registry name, Hugging Face id, or local path")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--field", default="text", help="JSONL field holding the text")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(args.tokenizer))
    total = write_token_file(iter_texts(args.inputs, args.field), tokenizer, args.out)
    print(f"wrote {total} tokens to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
