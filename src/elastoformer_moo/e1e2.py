"""E1 and E2: quality of every nested configuration before any training.

E1 compares, per configuration and per evaluation set, the model in its original weight order
(a naive prefix: whatever channels happen to come first) against the importance-ordered model.
E2 is the permutation itself: it must leave the full model's function unchanged, and the report
records by how much the logits moved.

No training happens here. The numbers are the starting point that joint nested training (E3) has
to beat, not a result of the method.
"""

from __future__ import annotations

import gc
import json
import platform
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from .config import make_grid
from .data import TokenBlocks
from .evaluate import evaluate_configs
from .nested import make_elastic
from .pipeline import load_causal_lm, permute_pretrained


def environment_info() -> dict:
    """Software and hardware facts that make a result interpretable later."""
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import flash_attn

        info["flash_attn"] = flash_attn.__version__
    except Exception as exc:  # not installed, or its extension cannot load
        info["flash_attn"] = f"unavailable ({type(exc).__name__})"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _evaluate_all(model_path: str, eval_sets: Mapping[str, str | Path], fractions, mlp_multiple, seq_len,
                  batch_size, max_eval_batches, dtype, device, attn_implementation) -> dict:
    lm = load_causal_lm(model_path, dtype, device, attn_implementation)
    ctrl = make_elastic(lm, mlp_multiple=mlp_multiple)
    configs = make_grid(ctrl.arch, fractions=fractions, mlp_multiple=mlp_multiple)
    out: dict = {}
    for name, path in eval_sets.items():
        blocks = TokenBlocks(path, seq_len)
        rows = evaluate_configs(lm, ctrl, configs, lambda: blocks.sequential(batch_size, max_eval_batches))
        full = rows[-1]["perplexity"]
        for row in rows:
            row["ppl_ratio_to_full"] = row["perplexity"] / full
        out[name] = rows
    del lm, ctrl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_e1_e2(
    model: str,
    out_dir: str | Path,
    calib_tokens: str | Path,
    eval_sets: Mapping[str, str | Path],
    seq_len: int = 1024,
    batch_size: int = 8,
    calib_batches: int = 16,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    mlp_multiple: int = 256,
    dtype: str = "bfloat16",
    attn_implementation: Optional[str] = None,
    device: str = "cpu",
    max_eval_batches: Optional[int] = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """Run E1 and E2 for one model and write ``results.json`` and ``summary.md`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "model": model,
        "protocol": {
            "seq_len": seq_len, "batch_size": batch_size, "dtype": dtype, "attn_implementation": attn_implementation,
            "calibration_tokens": str(calib_tokens), "calibration_blocks": calib_batches * batch_size,
            "eval_sets": {k: str(v) for k, v in eval_sets.items()}, "fractions": list(fractions),
            "mlp_multiple": mlp_multiple, "max_eval_batches": max_eval_batches,
        },
        "environment": environment_info(),
    }
    results["naive_prefix"] = _evaluate_all(
        model, eval_sets, fractions, mlp_multiple, seq_len, batch_size, max_eval_batches,
        dtype, device, attn_implementation,
    )

    # E2: permute in float32 (a low-precision equivalence check would mix rounding noise with real
    # errors), verify the logits are unchanged, and store the weights in the evaluation dtype.
    permuted_dir = out / "permuted"
    results["permutation"] = permute_pretrained(
        model, permuted_dir, calib_tokens, seq_len=seq_len, batch_size=batch_size, n_batches=calib_batches,
        compute_dtype="float32", save_dtype=dtype, atol=atol, rtol=rtol, device=device,
    )
    results["importance_ordered"] = _evaluate_all(
        str(permuted_dir), eval_sets, fractions, mlp_multiple, seq_len, batch_size, max_eval_batches,
        dtype, device, attn_implementation,
    )

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "summary.md").write_text(format_summary(results))
    return results


def format_summary(results: Mapping) -> str:
    """Markdown tables of perplexity per configuration, original order versus importance-ordered."""
    lines = [f"# E1/E2: {results['model']}", ""]
    proto, env = results["protocol"], results["environment"]
    lines += [
        f"dtype {proto['dtype']}, attention {proto['attn_implementation']}, seq_len {proto['seq_len']}, "
        f"torch {env['torch']}, transformers {env['transformers']}, flash_attn {env['flash_attn']}"
        + (f", GPU {env['gpu']}" if "gpu" in env else ""),
        "",
        "No training: these are the starting points that joint nested training has to beat.",
        "",
    ]
    perm = results["permutation"]
    lines += [f"Permutation check (E2): max logit difference {perm['max_logit_diff']:.3e} in "
              f"{perm['compute_dtype']} (atol {perm['atol']}, rtol {perm['rtol']}); "
              f"{perm['calibration_tokens']} calibration tokens.", ""]
    for name in results["naive_prefix"]:
        naive, ordered = results["naive_prefix"][name], results["importance_ordered"][name]
        lines += [f"## {name}", "", "| config | KV groups | MLP width | active params | ppl, original order | ppl, importance-ordered |",
                  "|---|---|---|---|---|---|"]
        for a, b in zip(naive, ordered):
            lines.append(f"| {a['config']} | {a['kv_groups']} | {a['mlp_width']} | {a['active_parameters']:,} | "
                         f"{a['perplexity']:.4g} | {b['perplexity']:.4g} |")
        lines.append("")
    return "\n".join(lines)
