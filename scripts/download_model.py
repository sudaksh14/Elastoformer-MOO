"""Download a model into the Hugging Face cache so jobs can run without network access.

    python scripts/download_model.py --model smollm2-360m
    HF_TOKEN=... python scripts/download_model.py --model llama3.2-1b     # gated: needs a token

The token is read from the HF_TOKEN environment variable only. It is never accepted as an argument
(command lines are visible to other users in `ps`) and never written to disk by this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.pipeline import resolve_model_path  # noqa: E402
from elastoformer_moo.registry import load_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="registry name or Hugging Face id")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo = resolve_model_path(args.model)
    spec = load_registry().get(args.model)
    token = os.environ.get("HF_TOKEN")
    if spec and spec.gated and not token:
        print(f"{repo} is gated. Accept its license on huggingface.co, then export HF_TOKEN (see README).", file=sys.stderr)
        return 3
    try:
        path = snapshot_download(
            repo, token=token,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
            ignore_patterns=["*.bin", "*.pth", "*.onnx", "*.gguf", "original/*"],
        )
    except GatedRepoError:
        print(f"access to {repo} was refused: accept the license on its model page and make sure HF_TOKEN is set.", file=sys.stderr)
        return 3
    except RepositoryNotFoundError:
        print(f"{repo} was not found (or needs a token to be seen).", file=sys.stderr)
        return 3
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"{repo} -> {path}  ({size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
