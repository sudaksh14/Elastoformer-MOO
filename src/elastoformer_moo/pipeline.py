"""File-level stages: permute a pretrained model, train nested configurations, evaluate them.

The scripts in ``scripts/`` are thin wrappers around these functions so the logic can be
tested on tiny models without a command line.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import yaml

from .arch import ArchInfo
from .config import ElasticConfig, make_grid
from .data import TokenBlocks
from .distributed import DistContext, barrier, device_for, init_distributed, shutdown
from .evaluate import evaluate_configs
from .cost import make_cost_fn
from .weights import ConfigWeights
from .importance import collect_importance
from .nested import make_elastic
from .permute import permutation_from_importance, verify_permutation_equivalence
from .registry import get_spec
from .train import TrainConfig, build_teacher, train

DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_model_path(model: str, registry_path: Optional[str] = None) -> str:
    """``model`` is a local directory, a registry name, or a Hugging Face id."""
    if Path(model).is_dir():
        return model
    try:
        return get_spec(model, registry_path).hf_id
    except KeyError:
        return model


ATTN_IMPLEMENTATIONS = ("eager", "sdpa", "flash_attention_2")


def load_causal_lm(
    model: str,
    dtype: str = "float32",
    device: str = "cpu",
    attn_implementation: Optional[str] = None,
    autocast: bool = False,
):
    """Load a causal LM.

    ``attn_implementation`` is passed through explicitly and never silently replaced: asking for
    ``flash_attention_2`` when it is unavailable raises instead of quietly using ``sdpa``.
    FlashAttention needs half-precision inputs, so it requires bfloat16/float16 weights unless
    ``autocast`` is True (float32 master weights run under bf16 autocast, as in training).
    """
    from transformers import AutoModelForCausalLM

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    if attn_implementation not in (None, *ATTN_IMPLEMENTATIONS):
        raise ValueError(f"attn_implementation must be one of {ATTN_IMPLEMENTATIONS}, got {attn_implementation!r}")
    if attn_implementation == "flash_attention_2" and DTYPES[dtype] not in (torch.bfloat16, torch.float16) and not autocast:
        raise ValueError(
            f"flash_attention_2 needs bfloat16 or float16 weights, got {dtype}; "
            "use a half-precision dtype, or float32 weights with bf16 autocast (autocast=True)"
        )
    path = resolve_model_path(model)
    token = os.environ.get("HF_TOKEN")
    kwargs = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPES[dtype], token=token, **kwargs)
    return loaded.to(device)


_SCIENTIFIC = re.compile(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+")


def _parse_value(raw: str) -> Any:
    """Parse a command-line value as YAML, but read ``2e-5`` as a float.

    PyYAML follows YAML 1.1, which only treats ``2.0e-5`` as a float and returns the
    string ``'2e-5'`` for the form people actually type.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _SCIENTIFIC.fullmatch(value.strip()):
        return float(value)
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested dict, in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {part!r} while applying {item!r}")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def permute_pretrained(
    model: str,
    out_dir: str | Path,
    calib_tokens: str | Path,
    seq_len: int = 512,
    batch_size: int = 4,
    n_batches: int = 32,
    compute_dtype: str = "float32",
    save_dtype: str = "bfloat16",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    device: str = "cpu",
    importance_method: str = "act_weight",
    skip_first_token: bool = False,
) -> dict:
    """Importance-order a pretrained model, verify the logits are unchanged, and save it.

    The permutation is applied and checked in ``compute_dtype`` (use float32: a low
    precision check would mix rounding noise with real errors), then the weights are cast to
    ``save_dtype`` for storage.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lm = load_causal_lm(model, compute_dtype, device)
    arch = ArchInfo.from_hf_config(lm.config)
    blocks = TokenBlocks(calib_tokens, seq_len)
    batches = list(blocks.sequential(batch_size, max_batches=n_batches))
    importance = collect_importance(lm, arch, batches, method=importance_method, skip_first_token=skip_first_token)
    perm = permutation_from_importance(importance)
    diff = verify_permutation_equivalence(lm, arch, perm, batches[0], atol=atol, rtol=rtol)

    lm.to(DTYPES[save_dtype]).save_pretrained(out)
    perm.save(str(out / "permutation.pt"))
    importance.save(str(out / "importance.pt"))
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(resolve_model_path(model)).save_pretrained(out)
    except Exception as exc:  # tokenizer is optional for the numeric pipeline
        print(f"warning: tokenizer not saved ({type(exc).__name__}: {exc})")
    report = {
        "source": model,
        "arch": asdict(arch),
        "calibration_tokens": int(sum(b.numel() for b in batches)),
        "max_logit_diff": diff,
        "atol": atol,
        "rtol": rtol,
        "compute_dtype": compute_dtype,
        "save_dtype": save_dtype,
    }
    (out / "permutation_report.json").write_text(json.dumps(report, indent=2))
    return report


def build_configs(arch: ArchInfo, spec: dict) -> list[ElasticConfig]:
    return make_grid(
        arch,
        fractions=spec.get("fractions", (0.25, 0.5, 0.75, 1.0)),
        mlp_multiple=spec.get("mlp_multiple", 256),
        min_kv_groups=spec.get("min_kv_groups", 1),
    )


def run_training(cfg: dict) -> dict:
    """Train nested configurations from a config dict (see configs/experiments).

    Under ``torchrun`` (``WORLD_SIZE`` > 1) every process runs this function: each loads the model onto its own
    GPU and trains on its own data, gradients are averaged once per optimizer step, and only rank 0 writes
    logs, checkpoints and results. ``data.batch_size`` is per process, so the global batch is
    ``world_size * batch_size * train.grad_accum`` sequences.
    """
    ctx = init_distributed()
    try:
        return _run_training(cfg, ctx)
    finally:
        shutdown(ctx)


def _run_training(cfg: dict, ctx: DistContext) -> dict:
    device = device_for(ctx, cfg.get("device"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lm = load_causal_lm(
        cfg["model"], cfg.get("model_dtype", "float32"), device,
        attn_implementation=cfg.get("attn"), autocast=bool(cfg.get("train", {}).get("autocast_bf16", False)),
    )
    ctrl = make_elastic(lm, mlp_multiple=cfg.get("elastic", {}).get("mlp_multiple", 256))
    configs = build_configs(ctrl.arch, cfg.get("elastic", {}))
    tcfg = TrainConfig(**cfg.get("train", {}))
    tcfg.validate()
    seq_len = cfg["data"]["seq_len"]
    cost_fn = make_cost_fn(tcfg.cost_axis, ctrl.arch, configs, seq_len)  # fails here, before any training, if unusable

    train_blocks = TokenBlocks(cfg["data"]["train_tokens"], cfg["data"]["seq_len"])
    batch_size = cfg["data"]["batch_size"]
    teacher = build_teacher(lm, ctrl) if tcfg.teacher == "frozen_copy" and tcfg.kd_weight > 0 else None

    if ctx.is_main:
        (out / "configs.json").write_text(json.dumps([asdict(c) for c in configs], indent=2))
        (out / "costs.json").write_text(json.dumps(
            {"axis": tcfg.cost_axis, "seq_len": seq_len, "cost": {c.name: cost_fn(c) for c in configs}}, indent=2))
        (out / "experiment.json").write_text(
            json.dumps({**cfg, "world_size": ctx.world_size, "global_batch_sequences":
                        ctx.world_size * batch_size * tcfg.grad_accum}, indent=2, default=str)
        )
    sampling = cfg["data"].get("sampling", "random")
    if sampling not in ("random", "epoch"):
        raise ValueError(f"data.sampling must be 'random' or 'epoch', got {sampling!r}")

    def make_batches(seed: int):
        if sampling == "epoch":
            # exact: no repeated blocks within an epoch, disjoint across ranks, and a resume skips what was consumed
            resumed_steps = seed - tcfg.seed
            return train_blocks.epoch_stream(
                batch_size, seed=tcfg.seed, rank=ctx.rank, world=ctx.world_size,
                skip_batches=resumed_steps * tcfg.grad_accum,
            )
        return train_blocks.random(batch_size, seed + 1_000_003 * ctx.rank)  # with replacement

    weights = ConfigWeights(configs, ctrl.arch, tcfg.weight_mode, tcfg.max_weight, tcfg.hv_w_min, tcfg.hv_w_max,
                            tcfg.hv_beta, tcfg.hv_ema, tcfg.hv_full_min)
    hv_batches = None
    if tcfg.weight_mode == "hv":
        # The outer step selects weights on held-out data (hv_val_set, default 'fineweb'). It is taken from the END
        # of that file, while the reported evaluation reads the START, so the two never overlap.
        val_cfg = cfg["data"]["val_tokens"]
        hv_name = tcfg.hv_val_set
        hv_path = val_cfg[hv_name] if isinstance(val_cfg, dict) else val_cfg
        hv_blocks = TokenBlocks(hv_path, cfg["data"]["seq_len"])
        eval_blocks = cfg["data"].get("eval_batches", 16) * batch_size  # the reported evaluation reads this prefix
        if len(hv_blocks) < eval_blocks + tcfg.hv_val_batches * batch_size:
            raise ValueError(f"{hv_path} has {len(hv_blocks)} blocks: too few for evaluation ({eval_blocks}) and HV selection")
        hv_cache = list(hv_blocks.tail(batch_size, tcfg.hv_val_batches))
        hv_batches = lambda: hv_cache  # noqa: E731  (the same fixed batches at every outer step)
    history = train(
        lm, ctrl, configs, make_batches, tcfg, out,
        teacher=teacher, resume=bool(cfg.get("resume", False)), dist_ctx=ctx,
        weights=weights, hv_val_batches=hv_batches, cost_fn=cost_fn,
    )
    if ctx.is_main:
        lm.save_pretrained(out / "final")
    barrier(ctx)

    result = {"final_step": history[-1]["step"] if history else None, "history_points": len(history),
              "world_size": ctx.world_size}
    val = cfg["data"].get("val_tokens")
    failure = None
    if val and ctx.is_main:
        # a failure here must not leave the other ranks waiting at the barrier until the NCCL timeout
        try:
            val_sets = {"val": val} if isinstance(val, (str, Path)) else dict(val)
            eval_batches = cfg["data"].get("eval_batches", 16)
            all_rows = {}
            for name, path in val_sets.items():
                blocks = TokenBlocks(path, cfg["data"]["seq_len"])
                all_rows[name] = evaluate_configs(
                    lm, ctrl, configs, lambda blocks=blocks: blocks.sequential(batch_size, max_batches=eval_batches),
                    autocast_bf16=tcfg.autocast_bf16,
                )
            by_name = {c.name: c for c in configs}
            for rows in all_rows.values():
                for row in rows:
                    row["cost"], row["cost_axis"] = cost_fn(by_name[row["config"]]), tcfg.cost_axis
            payload = all_rows["val"] if isinstance(val, (str, Path)) else all_rows
            (out / "eval.json").write_text(json.dumps(payload, indent=2))
            result["eval"] = payload
        except Exception as exc:  # re-raised after every rank has reached the barrier
            failure = exc
    barrier(ctx)
    if failure is not None:
        raise failure
    return result


def evaluate_pretrained(
    model: str,
    tokens: str | Path,
    seq_len: int,
    batch_size: int,
    fractions: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    mlp_multiple: int = 256,
    max_batches: Optional[int] = 16,
    dtype: str = "float32",
    device: str = "cpu",
    attn_implementation: Optional[str] = None,
) -> list[dict]:
    """Perplexity of each nested configuration of an already-loaded checkpoint (no training)."""
    lm = load_causal_lm(model, dtype, device, attn_implementation)
    ctrl = make_elastic(lm, mlp_multiple=mlp_multiple)
    configs = make_grid(ctrl.arch, fractions=fractions, mlp_multiple=mlp_multiple)
    blocks = TokenBlocks(tokens, seq_len)
    return evaluate_configs(lm, ctrl, configs, lambda: blocks.sequential(batch_size, max_batches=max_batches))
