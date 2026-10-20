"""Joint nested training: sandwich sampling, distillation, optional monotonicity penalty.

Every optimizer step draws a set of configurations (largest, smallest, random others),
runs each on the same shared weights, and sums their losses. No parameter is frozen.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import torch

from .config import ElasticConfig
from .distributed import (
    SINGLE,
    DistContext,
    assert_params_in_sync,
    barrier,
    gather_objects,
    reduce_mean,
    sync_gradients,
)
from .losses import distillation_kl, lm_cross_entropy, monotonicity_penalty
from .nested import ElasticController
from .sampling import SandwichSampler
from .weights import ConfigWeights, gradient_dots, hv_utilities

TEACHER_MODES = ("frozen_copy", "self_full", "none")
WEIGHT_MODES = ("fixed", "hv")


@dataclass
class TrainConfig:
    steps: int = 1000
    lr: float = 2e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum: int = 1
    ce_weight: float = 1.0
    kd_weight: float = 1.0
    temperature: float = 1.0
    mono_weight: float = 0.0
    n_random: int = 1
    weight_mode: str = "fixed"      # 'fixed' (max_weight on the largest config) or 'hv' (hypervolume-driven, see weights.py)
    max_weight: float = 1.0         # weight of the largest configuration; the others start at 1
    hv_interval: int = 50           # optimizer steps between outer (weight) updates in hv mode
    hv_beta: float = 0.5            # exponent scale of the multiplicative update
    hv_ema: float = 0.9             # smoothing of the utilities across updates
    hv_ref_scale: float = 1.25      # reference cost = hv_ref_scale * largest cost
    hv_w_min: float = 0.25
    hv_w_max: float = 4.0
    hv_full_min: float = 1.0        # the largest configuration's weight never drops below this
    hv_val_set: str = "fineweb"     # the validation set the HV outer step selects on (its last blocks are excluded from re-scoring)
    hv_val_batches: int = 2         # validation batches per outer update
    cost_axis: str = "params"   # what 'cost' means for reporting and learned weights: see cost.py
    sandwich: bool = True   # False: naive recipe, n_random configurations drawn uniformly from all of them
    teacher: str = "frozen_copy"
    autocast_bf16: bool = False
    seed: int = 0
    log_every: int = 10
    save_every: int = 0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            expected = _FIELD_TYPES[name]
            ok = isinstance(value, expected) and not isinstance(value, bool)
            if name in ("teacher", "cost_axis", "weight_mode", "hv_val_set"):
                ok = isinstance(value, str)
            elif name in ("autocast_bf16", "sandwich"):
                ok = isinstance(value, bool)
            elif name == "betas":
                ok = isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(b, (int, float)) for b in value)
            if not ok:
                raise TypeError(
                    f"TrainConfig.{name} must be {getattr(expected, '__name__', expected)}, got {value!r} "
                    f"({type(value).__name__}). In YAML write floats with a dot, e.g. 2.0e-5, not 2e-5."
                )
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {self.weight_mode!r}")
        if self.hv_interval < 1 or self.hv_val_batches < 1 or self.max_weight <= 0:
            raise ValueError("hv_interval and hv_val_batches must be at least 1 and max_weight positive")
        if self.teacher not in TEACHER_MODES:
            raise ValueError(f"teacher must be one of {TEACHER_MODES}, got {self.teacher!r}")
        if self.steps < 1 or self.grad_accum < 1:
            raise ValueError("steps and grad_accum must be at least 1")
        if self.kd_weight > 0 and self.teacher == "none":
            raise ValueError("kd_weight > 0 requires a teacher ('frozen_copy' or 'self_full')")


_FIELD_TYPES = {
    "steps": int, "lr": (int, float), "min_lr_ratio": (int, float), "warmup_steps": int,
    "weight_decay": (int, float), "betas": tuple, "grad_clip": (int, float), "grad_accum": int,
    "ce_weight": (int, float), "kd_weight": (int, float), "temperature": (int, float),
    "mono_weight": (int, float), "n_random": int, "sandwich": bool, "cost_axis": str, "weight_mode": str, "max_weight": (int, float), "hv_interval": int,
    "hv_beta": (int, float), "hv_ema": (int, float), "hv_ref_scale": (int, float), "hv_w_min": (int, float),
    "hv_w_max": (int, float), "hv_full_min": (int, float), "hv_val_set": str, "hv_val_batches": int, "teacher": str, "autocast_bf16": bool,
    "seed": int, "log_every": int, "save_every": int,
}


def build_teacher(model, controller: ElasticController):
    """Frozen copy of the current full model.

    The elastic patches are removed before copying: a deep copy would otherwise keep
    closures that point at the student's modules, and the "teacher" would run the student.
    """
    previous = controller.cfg
    controller.detach()
    try:
        teacher = copy.deepcopy(model)
    finally:
        controller.attach()
        controller.cfg = previous
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip = param.ndim < 2 or "norm" in name or "embed" in name
        (no_decay if skip else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_lambda(tcfg: TrainConfig) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < tcfg.warmup_steps:
            return (step + 1) / max(1, tcfg.warmup_steps)
        progress = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return tcfg.min_lr_ratio + (1 - tcfg.min_lr_ratio) * cosine

    return fn


def nested_micro_step(
    model,
    controller: ElasticController,
    configs: Sequence[ElasticConfig],
    input_ids: torch.Tensor,
    tcfg: TrainConfig,
    teacher=None,
    weights: Optional[dict] = None,
) -> dict:
    """Forward and backward over ``configs`` for one micro-batch. Gradients accumulate in ``model``.

    ``weights`` maps configuration name to loss weight (default 1); the loss is the weighted mean over ``configs``.
    """
    n = len(configs)
    w = {c.name: (weights or {}).get(c.name, 1.0) for c in configs}
    w_sum = sum(w.values())
    device_type = input_ids.device.type

    teacher_logits = None
    if tcfg.kd_weight > 0 and tcfg.teacher != "none":
        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=tcfg.autocast_bf16):
            if tcfg.teacher == "frozen_copy":
                teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
            else:  # self_full: current full config, detached
                with controller.use(None):
                    teacher_logits = model(input_ids=input_ids, use_cache=False).logits.detach()

    keep_graph = tcfg.mono_weight > 0 and n > 1
    metrics: dict = {"configs": {}}
    total, task_losses = None, []
    for cfg in configs:
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=tcfg.autocast_bf16):
            with controller.use(cfg):
                logits = model(input_ids=input_ids, use_cache=False).logits
        ce = lm_cross_entropy(logits, input_ids)
        kd = distillation_kl(logits, teacher_logits, tcfg.temperature) if teacher_logits is not None else ce.new_zeros(())
        loss = w[cfg.name] * (tcfg.ce_weight * ce + tcfg.kd_weight * kd) / w_sum
        metrics["configs"][cfg.name] = {"ce": ce.item(), "kd": kd.item()}
        if keep_graph:
            total = loss if total is None else total + loss
            task_losses.append(ce)
        else:
            (loss / tcfg.grad_accum).backward()
    if keep_graph:
        sizes = [c.size(controller.arch) for c in configs]
        penalty = monotonicity_penalty(task_losses, sizes)
        metrics["mono_penalty"] = penalty.item()
        ((total + tcfg.mono_weight * penalty) / tcfg.grad_accum).backward()
    metrics["loss"] = sum(v["ce"] * tcfg.ce_weight + v["kd"] * tcfg.kd_weight for v in metrics["configs"].values()) / n
    return metrics


def _save_state(path: Path, model, optimizer, scheduler, sampler_states, step: int, weights_state=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler_states,  # one state per rank, in rank order
            "step": step,
            "weights": weights_state,
        },
        tmp,
    )
    tmp.replace(path)


def _checkpoint(path: Path, model, optimizer, scheduler, sampler, step: int, ctx: DistContext, weights=None) -> None:
    """Every rank takes part (the sampler states are gathered); only the main rank writes."""
    states = gather_objects(sampler.state(), ctx)
    if ctx.is_main:
        _save_state(path, model, optimizer, scheduler, states, step, None if weights is None else weights.state())
    barrier(ctx)


def _hv_outer_step(model, controller, configs, train_batch, tcfg, teacher, weights, val_batches_fn, cost_fn, ctx, step) -> dict:
    """Outer step of the hypervolume-weighted recipe: measure gradient alignments, update the weights."""
    ordered = sorted(configs, key=lambda c: c.size(controller.arch))
    device = train_batch.device

    def loss_fn(m, ctrl, cfg, batch, training):
        batch = batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=tcfg.autocast_bf16):
            with ctrl.use(cfg):
                logits = m(input_ids=batch, use_cache=False).logits
            ce = lm_cross_entropy(logits, batch)
            if not training or tcfg.kd_weight <= 0 or tcfg.teacher == "none":
                return ce
            with torch.no_grad():
                if tcfg.teacher == "frozen_copy":
                    t_logits = teacher(input_ids=batch, use_cache=False).logits
                else:
                    with ctrl.use(None):
                        t_logits = m(input_ids=batch, use_cache=False).logits.detach()
        return tcfg.ce_weight * ce + tcfg.kd_weight * distillation_kl(logits, t_logits, tcfg.temperature)

    dots, val_losses = gradient_dots(model, controller, ordered, train_batch, list(val_batches_fn())[: tcfg.hv_val_batches],
                                     loss_fn, tcfg.autocast_bf16)
    gathered = gather_objects({"dots": dots, "val": val_losses}, ctx)
    n = len(ordered)
    dots = [[sum(g["dots"][i][j] for g in gathered) / len(gathered) for j in range(n)] for i in range(n)]
    val_losses = [sum(g["val"][i] for g in gathered) / len(gathered) for i in range(n)]
    costs = [cost_fn(c) for c in ordered]
    utilities = hv_utilities(dots, costs, val_losses, tcfg.hv_ref_scale)
    new = weights.update({c.name: u for c, u in zip(ordered, utilities)})
    return {"step": step, "configs": [c.name for c in ordered], "val_ce": val_losses, "costs": costs,
            "dots": dots, "utilities": utilities, "weights": new}


def train(
    model,
    controller: ElasticController,
    configs: Sequence[ElasticConfig],
    make_batches: Callable[[int], Iterator[torch.Tensor]],
    tcfg: TrainConfig,
    out_dir: str | Path,
    teacher=None,
    resume: bool = False,
    dist_ctx: DistContext = SINGLE,
    weights: Optional[ConfigWeights] = None,
    hv_val_batches: Optional[Callable[[], Sequence[torch.Tensor]]] = None,
    cost_fn: Optional[Callable[[ElasticConfig], float]] = None,
) -> list[dict]:
    """Run joint nested training and return the logged metrics.

    ``make_batches(seed)`` returns an iterator of ``[B, T]`` token batches. On resume it is
    called with ``tcfg.seed + step`` so the data stream is deterministic given
    ``(seed, step)``; it is not bit-identical to an uninterrupted run.

    With ``dist_ctx.world_size > 1`` every rank runs this function on its own data: ``make_batches`` must
    give each rank different batches, and each rank draws its own sandwich samples (seeded by rank).
    Gradients are averaged once per optimizer step, so the update equals one step on the combined batch.
    Only the main rank writes logs and checkpoints; every rank must call this with the same ``tcfg``.
    """
    tcfg.validate()
    if weights is not None and weights.mode == "hv" and (hv_val_batches is None or cost_fn is None):
        raise ValueError("weight_mode='hv' needs hv_val_batches and cost_fn")
    if tcfg.teacher == "frozen_copy" and tcfg.kd_weight > 0 and teacher is None:
        raise ValueError("teacher='frozen_copy' needs a teacher model; build one with build_teacher()")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    sampler = SandwichSampler(configs, controller.arch, tcfg.n_random, tcfg.seed + 7919 * dist_ctx.rank, sandwich=tcfg.sandwich)
    optimizer = torch.optim.AdamW(_param_groups(model, tcfg.weight_decay), lr=tcfg.lr, betas=tcfg.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(tcfg))

    step = 0
    state_path = out / "trainer_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        saved = state["sampler"]
        if isinstance(saved, list) and len(saved) == dist_ctx.world_size:
            sampler.set_state(saved[dist_ctx.rank])
        step = state["step"]
        if weights is not None and state.get("weights") is not None:
            weights.set_state(state["weights"])
        del state
    if dist_ctx.is_main:
        (out / "train_config.json").write_text(json.dumps(asdict(tcfg), indent=2))
    barrier(dist_ctx)

    batches = make_batches(tcfg.seed + step)
    log_path = out / "metrics.jsonl"
    history: list[dict] = []
    model.train()
    start = time.time()
    while step < tcfg.steps:
        optimizer.zero_grad(set_to_none=True)
        step_metrics = None
        for _ in range(tcfg.grad_accum):
            batch = next(batches).to(device)
            step_metrics = nested_micro_step(
                model, controller, sampler.sample(), batch, tcfg, teacher, None if weights is None else weights.as_dict())
        sync_gradients(model.parameters(), dist_ctx)  # one average per optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip) if tcfg.grad_clip > 0 else None
        optimizer.step()
        scheduler.step()
        step += 1

        hv_record = None
        if weights is not None and weights.mode == "hv" and step % tcfg.hv_interval == 0 and step < tcfg.steps:
            hv_record = _hv_outer_step(model, controller, configs, batch, tcfg, teacher, weights, hv_val_batches, cost_fn,
                                       dist_ctx, step)
            if dist_ctx.is_main:
                with open(out / "hv_log.jsonl", "a") as handle:
                    handle.write(json.dumps(hv_record) + "\n")
        if step % tcfg.log_every == 0 or step == tcfg.steps:
            step_metrics["loss"] = reduce_mean(step_metrics["loss"], dist_ctx, device)
            assert_params_in_sync(model, dist_ctx)
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": None if grad_norm is None else float(grad_norm),
                "elapsed_s": round(time.time() - start, 2),
                "world_size": dist_ctx.world_size,
                **step_metrics,
            }
            if weights is not None:
                record["weights"] = weights.as_dict()
            if device.type == "cuda":
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            history.append(record)
            if dist_ctx.is_main:
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
        if tcfg.save_every and step % tcfg.save_every == 0 and step < tcfg.steps:
            _checkpoint(state_path, model, optimizer, scheduler, sampler, step, dist_ctx, weights)
    _checkpoint(state_path, model, optimizer, scheduler, sampler, step, dist_ctx, weights)
    return history
