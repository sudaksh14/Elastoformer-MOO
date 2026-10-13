"""Data-parallel training across processes, with explicit gradient averaging.

Why not DistributedDataParallel: each optimizer step runs several backward passes (one per sampled
configuration) through modules whose ``forward`` is patched with closures, and DDP's per-backward hooks
would all-reduce after every one of them. Here each rank accumulates all of its configurations' gradients
locally, and one bucketed all-reduce averages them once per optimizer step. Every rank then applies the
same averaged gradient with the same optimizer, so the weights stay identical on all ranks.

Launch with ``torchrun`` (or ``python -m torch.distributed.run``); a single process needs no setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: Optional[str] = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


SINGLE = DistContext()


def init_distributed(backend: Optional[str] = None) -> DistContext:
    """Initialize from the environment variables ``torchrun`` sets; a plain run returns ``SINGLE``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return SINGLE
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    # rank 0 alone runs the final evaluation while the others wait at a barrier, so the default 10-minute
    # timeout is too short for a long one
    minutes = int(os.environ.get("ELASTOFORMER_DIST_TIMEOUT_MIN", "90"))
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=minutes))
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank, backend=backend)


def device_for(ctx: DistContext, requested: Optional[str] = None) -> torch.device:
    """The device this rank trains on: its own GPU under NCCL, else the requested device or CPU."""
    if requested in (None, "cuda") and torch.cuda.is_available():
        return torch.device("cuda", ctx.local_rank if ctx.distributed else 0)
    return torch.device(requested or "cpu")


def barrier(ctx: DistContext) -> None:
    if ctx.distributed:
        dist.barrier()


def shutdown(ctx: DistContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def gather_objects(obj, ctx: DistContext) -> list:
    """Every rank's ``obj``, in rank order, on every rank."""
    if not ctx.distributed:
        return [obj]
    out = [None] * ctx.world_size
    dist.all_gather_object(out, obj)
    return out


def reduce_mean(value: float, ctx: DistContext, device: torch.device) -> float:
    if not ctx.distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / ctx.world_size).item()


@torch.no_grad()
def sync_gradients(params: Iterable[torch.nn.Parameter], ctx: DistContext, bucket_elements: int = 2**27) -> None:
    """Average ``.grad`` across ranks in place, in buckets of at most ``bucket_elements`` values.

    A parameter with no gradient on this rank is treated as zero so every rank issues the same
    collectives in the same order (a mismatch would hang the job).
    """
    if not ctx.distributed:
        return
    params = [p for p in params if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    bucket: list[torch.nn.Parameter] = []
    size = 0

    def flush() -> None:
        nonlocal bucket, size
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= ctx.world_size
        offset = 0
        for p in bucket:
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n
        bucket, size = [], 0

    last_key = None
    for p in params:
        key = (p.grad.dtype, p.grad.device)
        if bucket and (key != last_key or size + p.numel() > bucket_elements):
            flush()
        bucket.append(p)
        size += p.numel()
        last_key = key
    flush()


@torch.no_grad()
def assert_params_in_sync(model: torch.nn.Module, ctx: DistContext, rtol: float = 1e-6) -> None:
    """Raise if the weights differ across ranks (a diverged replica silently ruins a run)."""
    if not ctx.distributed:
        return
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for p in model.parameters():
        total[0] += p.double().sum()
        total[1] += p.double().abs().sum()
    gathered = [torch.zeros_like(total) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, total)
    reference = gathered[0]
    for rank, other in enumerate(gathered):
        if not torch.allclose(other, reference, rtol=rtol, atol=0.0):
            raise RuntimeError(
                f"model replicas diverged: rank {rank} checksum {other.tolist()} vs rank 0 {reference.tolist()}"
            )
