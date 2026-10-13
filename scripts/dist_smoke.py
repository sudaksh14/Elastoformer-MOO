"""Multi-GPU smoke test. Run under torchrun on a node with at least two GPUs:

    python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/dist_smoke.py

Checks, on real GPUs with NCCL: collective correctness, gradient averaging, a few nested training
steps under bf16 autocast with every rank on its own data, that the weights stay identical across
ranks, and the all-reduce bandwidth for a gradient buffer the size of SmolLM2-360M. Exits non-zero
on any failure; exits 2 if it was not launched with several processes or CUDA is unavailable.
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.distributed import (  # noqa: E402
    assert_params_in_sync,
    barrier,
    device_for,
    init_distributed,
    shutdown,
    sync_gradients,
)
from elastoformer_moo.nested import make_elastic  # noqa: E402
from elastoformer_moo.train import TrainConfig, build_teacher, train  # noqa: E402


def main() -> int:
    ctx = init_distributed()
    if not ctx.distributed or not torch.cuda.is_available():
        print("run under torchrun with at least 2 processes on a GPU node", file=sys.stderr)
        return 2
    device = device_for(ctx)
    report = {"world_size": ctx.world_size, "backend": ctx.backend, "device": torch.cuda.get_device_name(device)}

    # 1. all-reduce correctness
    t = torch.full((4096,), float(ctx.rank + 1), device=device)
    dist.all_reduce(t)
    expected = ctx.world_size * (ctx.world_size + 1) / 2
    assert torch.all(t == expected), f"all_reduce gave {t[0].item()}, expected {expected}"

    # 2. gradient averaging
    lin = torch.nn.Linear(8, 8).to(device)
    for p in lin.parameters():
        p.grad = torch.full_like(p, float(ctx.rank))
    sync_gradients(lin.parameters(), ctx)
    mean = (ctx.world_size - 1) / 2
    assert all(torch.allclose(p.grad, torch.full_like(p, mean)) for p in lin.parameters()), "gradient mean is wrong"

    # 3. nested training, every rank on its own data
    torch.manual_seed(0)  # identical initial weights on every rank
    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=384, num_hidden_layers=4, num_attention_heads=8,
                    num_key_value_heads=4, head_dim=16, max_position_embeddings=256, tie_word_embeddings=False)
    ).to(device)
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=32)
    teacher = build_teacher(model, ctrl)

    def make_batches(seed):
        g = torch.Generator().manual_seed(seed * 100 + ctx.rank)
        for _ in itertools.count():
            yield torch.randint(0, 1024, (4, 64), generator=g)

    tcfg = TrainConfig(steps=6, lr=1e-3, warmup_steps=1, kd_weight=1.0, autocast_bf16=True, log_every=2)
    out = Path(tempfile.mkdtemp(prefix="dist_smoke_")) if ctx.is_main else None
    shared = [str(out) if out else None]
    dist.broadcast_object_list(shared, src=0)
    history = train(model, ctrl, configs, make_batches, tcfg, shared[0], teacher=teacher, dist_ctx=ctx)
    assert_params_in_sync(model, ctx)
    losses = [h["loss"] for h in history]
    assert all(l == l and abs(l) < 1e6 for l in losses), f"non-finite loss {losses}"
    report["nested_training_losses"] = losses

    # 4. all-reduce bandwidth for a SmolLM2-360M-sized fp32 gradient buffer
    buf = torch.ones(362_000_000 // 1, dtype=torch.float32, device=device)
    for _ in range(2):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    barrier(ctx)
    start = time.time()
    reps = 5
    for _ in range(reps):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    seconds = (time.time() - start) / reps
    report["allreduce_1.45GB_seconds"] = round(seconds, 4)
    report["allreduce_algbw_GBps"] = round(buf.numel() * 4 / 1e9 / seconds, 2)
    report["status"] = "ok"

    if ctx.is_main:
        print(json.dumps(report))
    barrier(ctx)
    shutdown(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
