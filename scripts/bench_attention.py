"""Compare attention backends on the elastic forward and backward pass (needs a GPU).

Answers an empirical question rather than assuming it: PyTorch's `sdpa` already dispatches to a
FlashAttention kernel when its conditions are met (half precision, a supported GPU), so the standalone
`flash_attn` package is not automatically faster. This measures both on the same weights.

For each backend and each configuration it reports forward time, forward+backward time and peak
memory, plus how far the logits differ from `sdpa`.

    python scripts/bench_attention.py --model smollm2-360m --out results/bench_attention.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.e1e2 import environment_info  # noqa: E402
from elastoformer_moo.losses import lm_cross_entropy  # noqa: E402
from elastoformer_moo.nested import make_elastic  # noqa: E402
from elastoformer_moo.pipeline import load_causal_lm  # noqa: E402


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="smollm2-360m")
    parser.add_argument("--backends", nargs="+", default=["sdpa", "flash_attention_2"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mlp-multiple", type=int, default=256)
    parser.add_argument("--tokens", default=None,
                        help="token file of real text for the accuracy check (random ids make predictions near-tied "
                             "and exaggerate bf16 differences, so use real text)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; run this on a GPU node.", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    report = {"environment": environment_info(), "model": args.model, "batch_size": args.batch_size,
              "seq_len": args.seq_len, "dtype": "bfloat16", "inputs": "real text" if args.tokens else "random ids",
              "backends": {}}
    reference_logits: dict[str, torch.Tensor] = {}

    # Ground truth for accuracy: the same weights in float32 with sdpa. Each bf16 backend is judged by how
    # close it gets to this, not by how close it gets to another bf16 backend.
    ref = load_causal_lm(args.model, "float32", "cuda", attn_implementation="sdpa")
    ref_ctrl = make_elastic(ref, mlp_multiple=args.mlp_multiple)
    configs = make_grid(ref_ctrl.arch, fractions=args.fractions, mlp_multiple=args.mlp_multiple)
    if args.tokens:
        from elastoformer_moo.data import TokenBlocks

        ids = next(TokenBlocks(args.tokens, args.seq_len).sequential(args.batch_size)).cuda()
    else:
        ids = torch.randint(0, ref.config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    fp32_logits: dict[str, torch.Tensor] = {}
    for cfg in configs:
        with torch.no_grad(), ref_ctrl.use(cfg):
            # kept on the CPU: ~0.8 GB per configuration would otherwise inflate the GPU peak-memory measurement
            fp32_logits[cfg.name] = ref(input_ids=ids, use_cache=False).logits.float().cpu()
    del ref, ref_ctrl
    torch.cuda.empty_cache()

    for backend in args.backends:
        # bf16 weights: the same dtype for every backend so the comparison isolates the attention kernel
        lm = load_causal_lm(args.model, "bfloat16", "cuda", attn_implementation=backend)
        ctrl = make_elastic(lm, mlp_multiple=args.mlp_multiple)
        rows = []
        for cfg in configs:
            with ctrl.use(cfg):
                lm.eval()
                with torch.no_grad():
                    logits = lm(input_ids=ids, use_cache=False).logits.float()
                    forward_ms = timed(lambda: lm(input_ids=ids, use_cache=False), args.warmup, args.iters)
                lm.train()

                def step():
                    lm.zero_grad(set_to_none=True)
                    lm_cross_entropy(lm(input_ids=ids, use_cache=False).logits, ids).backward()

                torch.cuda.reset_peak_memory_stats()
                train_ms = timed(step, args.warmup, args.iters)
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
            reference = fp32_logits[cfg.name]
            logits = logits.cpu()
            err = (logits - reference).abs()
            row = {"config": cfg.name, "kv_groups": cfg.kv_groups, "mlp_width": cfg.mlp_width,
                   "forward_ms": forward_ms, "forward_backward_ms": train_ms, "peak_memory_gb": peak_gb,
                   "max_abs_err_vs_fp32": err.max().item(), "mean_abs_err_vs_fp32": err.mean().item(),
                   "max_abs_logit_in_fp32": reference.abs().max().item(),
                   "top1_agreement_vs_fp32": (logits.argmax(-1) == reference.argmax(-1)).float().mean().item()}
            if backend == "sdpa":
                reference_logits[cfg.name] = logits
            elif cfg.name in reference_logits:
                diff = (logits - reference_logits[cfg.name]).abs()
                row["max_abs_logit_diff_vs_sdpa"] = diff.max().item()
                row["top1_agreement_vs_sdpa"] = (logits.argmax(-1) == reference_logits[cfg.name].argmax(-1)).float().mean().item()
            rows.append(row)
            print(backend, row, flush=True)
        report["backends"][backend] = rows
        del lm, ctrl
        torch.cuda.empty_cache()

    if "sdpa" in report["backends"] and "flash_attention_2" in report["backends"]:
        for a, b in zip(report["backends"]["sdpa"], report["backends"]["flash_attention_2"]):
            b["speedup_forward_vs_sdpa"] = a["forward_ms"] / b["forward_ms"]
            b["speedup_forward_backward_vs_sdpa"] = a["forward_backward_ms"] / b["forward_backward_ms"]
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
