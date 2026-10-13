"""E0: fetch each registry model's config.json and report the facts the elastic design depends on.

Network access is required (and HF_TOKEN for gated models). Nothing is cached in the repo.

    python scripts/check_models.py --out artifacts/model_facts.json
    python scripts/check_models.py --only llama3.2-1b qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elastoformer_moo.arch import ArchInfo, UnsupportedArchitectureError  # noqa: E402
from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.registry import load_registry  # noqa: E402


def facts_from_config(config) -> dict:
    """Elastic-design facts for one Hugging Face config object."""
    out = {"model_type": getattr(config, "model_type", None)}
    try:
        arch = ArchInfo.from_hf_config(config)
    except UnsupportedArchitectureError as exc:
        out["supported"] = False
        out["reason"] = str(exc)
        return out
    out.update(
        supported=True,
        layers=arch.num_layers,
        hidden=arch.hidden_size,
        heads=arch.num_heads,
        kv_heads=arch.num_kv_heads,
        head_dim=arch.head_dim,
        intermediate=arch.intermediate_size,
        tied_embeddings=arch.tie_word_embeddings,
        vocab=getattr(config, "vocab_size", None),
        grid=[(c.name, c.kv_groups, c.mlp_width) for c in make_grid(arch)],
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="registry names to check")
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from transformers import AutoConfig

    token = os.environ.get("HF_TOKEN")
    api = HfApi()
    results = {}
    for name, spec in load_registry(args.registry).items():
        if args.only and name not in args.only:
            continue
        row = {"hf_id": spec.hf_id, "expected_model_type": spec.model_type}
        try:
            config = AutoConfig.from_pretrained(spec.hf_id, token=token)
            row.update(facts_from_config(config))
            row["model_type_matches_registry"] = row["model_type"] == spec.model_type
        except Exception as exc:  # gated, missing, offline, unknown architecture
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        try:
            info = api.model_info(spec.hf_id, token=token)
            row["license"] = getattr(info.card_data, "license", None) if info.card_data else None
        except Exception as exc:
            row["license_error"] = f"{type(exc).__name__}"
        results[name] = row
        print(f"{name:20s} {row.get('model_type')!s:8s} " + (row.get("error") or json.dumps({
            k: row[k] for k in ("layers", "hidden", "heads", "kv_heads", "head_dim", "intermediate", "tied_embeddings") if k in row
        })))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
