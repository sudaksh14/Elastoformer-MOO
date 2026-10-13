"""Multi-process training, tested on CPU with the gloo backend (two real processes).

The central check: training on two ranks, each seeing half of every batch, must produce the same
weights as one process seeing the whole batch. That is what makes the gradient averaging correct and
not merely plausible.
"""

import itertools
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from elastoformer_moo.config import make_grid
from elastoformer_moo.distributed import (
    SINGLE,
    DistContext,
    assert_params_in_sync,
    gather_objects,
    reduce_mean,
    sync_gradients,
)
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.train import TrainConfig, build_teacher, train

from conftest import tiny_model

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="torch.distributed with gloo is unavailable"
)


def _file_url(path: Path) -> str:
    return "file:///" + path.as_posix().lstrip("/")


def _join(rank: int, world: int, store: Path) -> DistContext:
    dist.init_process_group("gloo", init_method=_file_url(store), rank=rank, world_size=world)
    return DistContext(rank=rank, world_size=world, local_rank=rank, backend="gloo")


def _spawn(fn, world: int, out_dir: Path, *extra):
    store = out_dir / "store"
    if store.exists():
        store.unlink()
    mp.spawn(fn, args=(world, store, str(out_dir), *extra), nprocs=world, join=True)


# --------------------------------------------------------------------------------------- collectives


def _worker_collectives(rank, world, store, out_dir):
    ctx = _join(rank, world, Path(store))
    result = {}

    model = torch.nn.Linear(4, 3)
    extra = torch.nn.Parameter(torch.zeros(5))
    for p in model.parameters():
        p.grad = torch.full_like(p, float(rank + 1))  # rank 0 -> 1.0, rank 1 -> 2.0
    extra.grad = torch.full_like(extra, 3.0) if rank == 0 else None  # None on one rank counts as zero
    sync_gradients([*model.parameters(), extra], ctx, bucket_elements=7)  # tiny buckets force several all-reduces
    result["grads_mean"] = [p.grad.flatten().tolist() for p in model.parameters()]
    result["extra_grad"] = extra.grad.tolist()

    result["reduced"] = reduce_mean(float(rank), ctx, torch.device("cpu"))
    result["gathered"] = gather_objects({"rank": rank}, ctx)

    synced = torch.nn.Linear(2, 2)
    torch.manual_seed(0)
    for p in synced.parameters():
        p.data.normal_()
    assert_params_in_sync(synced, ctx)  # identical on both ranks: no error
    if rank == 1:
        synced.weight.data += 1.0
    try:
        assert_params_in_sync(synced, ctx)
        result["divergence_detected"] = False
    except RuntimeError as exc:
        result["divergence_detected"] = "diverged" in str(exc)

    Path(out_dir, f"collectives_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def test_gradient_averaging_and_collectives(tmp_path):
    _spawn(_worker_collectives, 2, tmp_path)
    for rank in (0, 1):
        r = json.loads((tmp_path / f"collectives_{rank}.json").read_text())
        assert all(v == pytest.approx(1.5) for flat in r["grads_mean"] for v in flat)  # (1 + 2) / 2
        assert r["extra_grad"] == pytest.approx([1.5] * 5)  # (3 + 0) / 2
        assert r["reduced"] == pytest.approx(0.5)
        assert r["gathered"] == [{"rank": 0}, {"rank": 1}]
        assert r["divergence_detected"] is True


def test_single_process_collectives_are_no_ops():
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.full((3,), 2.0)
    sync_gradients([p], SINGLE)
    assert p.grad.tolist() == [2.0, 2.0, 2.0]
    assert reduce_mean(5.0, SINGLE, torch.device("cpu")) == 5.0
    assert gather_objects("x", SINGLE) == ["x"]
    assert_params_in_sync(torch.nn.Linear(2, 2), SINGLE)


# ----------------------------------------------------------------------------- training equivalence


def _train_run(rank, world, store, out_dir, steps, ctx):
    model = tiny_model("llama", seed=0)
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 1.0), mlp_multiple=16)
    teacher = build_teacher(model, ctrl)
    data = torch.randint(0, 128, (steps, 4, 12), generator=torch.Generator().manual_seed(123))
    per_rank = data.shape[1] // world

    def make_batches(seed):
        for s in itertools.count():
            yield data[s % steps][rank * per_rank : (rank + 1) * per_rank]

    # n_random=0: every step uses exactly the largest and smallest configuration on every rank, so the
    # one-process and two-process runs see identical configurations and differ only in how data is split
    tcfg = TrainConfig(steps=steps, lr=1e-3, warmup_steps=1, n_random=0, kd_weight=1.0, grad_clip=1.0, log_every=1)
    history = train(model, ctrl, configs, make_batches, tcfg, Path(out_dir) / f"run_w{world}", teacher=teacher, dist_ctx=ctx)
    return model, history


def _worker_train(rank, world, store, out_dir, steps):
    ctx = _join(rank, world, Path(store))
    model, history = _train_run(rank, world, store, out_dir, steps, ctx)
    if rank == 0:
        torch.save(model.state_dict(), Path(out_dir) / f"final_w{world}.pt")
        (Path(out_dir) / f"history_w{world}.json").write_text(json.dumps(history))
    dist.destroy_process_group()


def test_two_ranks_reproduce_one_process_on_the_combined_batch(tmp_path):
    steps = 3
    single_model, single_history = _train_run(0, 1, None, str(tmp_path), steps, SINGLE)
    _spawn(_worker_train, 2, tmp_path, steps)

    two = torch.load(tmp_path / "final_w2.pt", map_location="cpu")
    # Adam normalizes each gradient element, which amplifies float32 rounding differences in rarely-updated
    # rows (embeddings) to about 1e-5; a wrong data split or a missing average moves weights by about lr = 1e-3.
    for name, tensor in single_model.state_dict().items():
        torch.testing.assert_close(two[name], tensor, atol=2e-5, rtol=1e-3, msg=f"parameter {name} differs")
    two_history = json.loads((tmp_path / "history_w2.json").read_text())
    assert [h["step"] for h in two_history] == [1, 2, 3]
    assert all(h["world_size"] == 2 for h in two_history)
    # the logged loss is the mean over ranks, so it matches the single-process loss on the same combined batch
    for a, b in zip(single_history, two_history):
        assert b["loss"] == pytest.approx(a["loss"], rel=1e-4)
        # Adam is invariant to gradient scale, so the weights alone cannot tell a mean from a sum. The
        # pre-clip gradient norm can: a summed gradient would be about twice the single-process norm.
        assert b["grad_norm"] == pytest.approx(a["grad_norm"], rel=1e-3)
    # only the main rank writes logs: exactly one line per logged step
    assert len((tmp_path / "run_w2" / "metrics.jsonl").read_text().splitlines()) == steps


# ------------------------------------------------------------------------------------------- resume


def _worker_resume(rank, world, store, out_dir):
    ctx = _join(rank, world, Path(store))
    model = tiny_model("llama", seed=0)
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16)
    teacher = build_teacher(model, ctrl)

    def make_batches(seed):
        g = torch.Generator().manual_seed(seed * 10 + rank)
        while True:
            yield torch.randint(0, 128, (2, 12), generator=g)

    run = Path(out_dir) / "resume_run"
    first = TrainConfig(steps=2, lr=1e-3, warmup_steps=1, kd_weight=1.0, log_every=1)
    train(model, ctrl, configs, make_batches, first, run, teacher=teacher, dist_ctx=ctx)
    longer = TrainConfig(steps=4, lr=1e-3, warmup_steps=1, kd_weight=1.0, log_every=1)
    history = train(model, ctrl, configs, make_batches, longer, run, teacher=teacher, resume=True, dist_ctx=ctx)
    assert_params_in_sync(model, ctx)  # raises if the ranks resumed into different weights
    state = torch.load(run / "trainer_state.pt", map_location="cpu", weights_only=False)
    Path(out_dir, f"resume_{rank}.json").write_text(json.dumps({
        "steps": [h["step"] for h in history], "saved_step": state["step"], "sampler_states": len(state["sampler"]),
    }))
    dist.destroy_process_group()


def test_resume_across_ranks_restores_step_and_per_rank_sampler_state(tmp_path):
    _spawn(_worker_resume, 2, tmp_path)
    for rank in (0, 1):
        r = json.loads((tmp_path / f"resume_{rank}.json").read_text())
        assert r["steps"] == [3, 4] and r["saved_step"] == 4
        assert r["sampler_states"] == 2  # one saved state per rank


# ------------------------------------------------------------------- run_training under torchrun-style env


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_run_training(rank, world, store, out_dir, port, cfg_json):
    # the environment torchrun sets for every worker process
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": str(rank),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "USE_LIBUV": "0"})  # the Windows torch build used for local tests has no libuv; harmless elsewhere
    from elastoformer_moo.pipeline import run_training

    result = run_training(json.loads(cfg_json))
    Path(out_dir, f"result_{rank}.json").write_text(json.dumps({k: v for k, v in result.items() if k != "eval"}))


def test_run_training_as_two_torchrun_style_processes(tmp_path):
    import numpy as np

    ckpt = tmp_path / "tiny"
    tiny_model("llama").save_pretrained(ckpt)
    rng = np.random.default_rng(0)
    for name in ("train", "val_a", "val_b"):
        rng.integers(0, 128, size=3000).astype(np.uint32).tofile(tmp_path / f"{name}.bin")
    cfg = {
        "model": str(ckpt), "device": "cpu", "out_dir": str(tmp_path / "run"),
        "elastic": {"fractions": [0.5, 0.75, 1.0], "mlp_multiple": 16},
        "train": {"steps": 3, "lr": 1e-3, "warmup_steps": 1, "log_every": 1, "grad_accum": 2},
        "data": {"train_tokens": str(tmp_path / "train.bin"),
                 "val_tokens": {"a": str(tmp_path / "val_a.bin"), "b": str(tmp_path / "val_b.bin")},
                 "seq_len": 16, "batch_size": 2, "eval_batches": 2},
    }
    store = tmp_path / "unused"
    mp.spawn(_worker_run_training, args=(2, store, str(tmp_path), _free_port(), json.dumps(cfg)), nprocs=2, join=True)

    out = tmp_path / "run"
    assert (out / "final" / "config.json").exists()
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 3  # one writer, not two
    experiment = json.loads((out / "experiment.json").read_text())
    assert experiment["world_size"] == 2 and experiment["global_batch_sequences"] == 2 * 2 * 2
    eval_rows = json.loads((out / "eval.json").read_text())
    assert set(eval_rows) == {"a", "b"} and all(len(rows) == 3 for rows in eval_rows.values())
    for rank in (0, 1):
        assert json.loads((tmp_path / f"result_{rank}.json").read_text())["final_step"] == 3
