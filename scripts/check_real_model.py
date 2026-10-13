"""Check the elastic code on a real checkpoint (no training, CPU is fine for small models).

Reports, for one model:
  1. the full configuration is bit-identical to the original model;
  2. every smaller configuration equals a physically sliced standalone model, in float64
     with RMSNorm kept in float64 (a difference near 1e-12 means the same function, not just
     a close one), and in float32 as relative error and top-1 agreement;
  3. importance ordering preserves the logits, and held-out perplexity of each configuration
     before and after it (no training; a smoke check of the mechanism, not an experiment).

    python scripts/check_real_model.py --model smollm2-135m --text README.md research/*.md \
        --out results/real_model_check_smollm2-135m.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.evaluate import perplexity  # noqa: E402
from elastoformer_moo.export import materialize  # noqa: E402
from elastoformer_moo.importance import collect_importance  # noqa: E402
from elastoformer_moo.nested import make_elastic  # noqa: E402
from elastoformer_moo.permute import permutation_from_importance, verify_permutation_equivalence  # noqa: E402
from elastoformer_moo.pipeline import load_causal_lm, resolve_model_path  # noqa: E402


def native_dtype_rmsnorm(model_type: str):
    """Patch the family's RMSNorm to stay in the input dtype. Returns an undo function."""
    module = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    norm_cls = next(getattr(module, n) for n in dir(module) if n.endswith("RMSNorm"))
    original = norm_cls.forward

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    norm_cls.forward = forward
    return lambda: setattr(norm_cls, "forward", original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="smollm2-135m")
    parser.add_argument("--text", nargs="+", required=True, help="text files used for calibration and held-out data")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mlp-multiple", type=int, default=64)
    parser.add_argument("--skip-float64", action="store_true", help="skip the float64 exactness check (8 bytes/param)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    text = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in args.text)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // args.seq_len) * args.seq_len
    blocks = ids[:n].view(-1, args.seq_len)
    calib, held = blocks[: len(blocks) // 2], blocks[len(blocks) // 2 :]
    if len(calib) < 2 or len(held) < 2:
        print("need at least 4 blocks of text; pass more --text files or lower --seq-len", file=sys.stderr)
        return 2
    pairs = lambda t: [t[i : i + 2] for i in range(0, len(t), 2)]  # noqa: E731

    report = {"model": args.model, "hf_path": path, "tokens": int(ids.numel()), "held_out_tokens": int(held.numel())}
    model = load_causal_lm(path, "float32")
    ctrl = make_elastic(model)
    arch = ctrl.arch
    configs = make_grid(arch, args.fractions, mlp_multiple=args.mlp_multiple)
    report["arch"] = arch.__dict__
    report["configs"] = [(c.name, c.kv_groups, c.mlp_width) for c in configs]
    batch = held[:2]

    # 1. full configuration is the original model
    with torch.no_grad():
        original = model(input_ids=batch).logits
        with ctrl.use(configs[-1]):
            report["full_config_bit_identical"] = bool(torch.equal(model(input_ids=batch).logits, original))

    # 2a. float32: relative error and top-1 agreement against the sliced standalone model
    fp32 = {}
    for cfg in configs[:-1]:
        small = materialize(model, cfg)
        with torch.no_grad():
            want = small(input_ids=batch).logits
            with ctrl.use(cfg):
                got = model(input_ids=batch).logits
        diff = (want - got).abs().max().item()
        fp32[cfg.name] = {
            "max_abs_diff": diff,
            "relative": diff / want.abs().max().item(),
            "top1_agreement": (want.argmax(-1) == got.argmax(-1)).float().mean().item(),
            "active_parameters": ctrl.active_parameters(cfg),
            "standalone_parameters": sum(p.numel() for p in small.parameters()),
        }
    report["float32_vs_sliced"] = fp32

    # 2b. float64 with RMSNorm in float64: exact equality is expected
    if args.skip_float64:
        report["float64_max_abs_diff_vs_sliced"] = "skipped"
    else:
        undo = native_dtype_rmsnorm(arch.model_type)
        try:
            # A fresh load, not a copy: deep-copying a model that has elastic patches attached would
            # keep closures pointing at the original modules.
            model64 = load_causal_lm(path, "float64")
            ctrl64 = make_elastic(model64)
            exact = {}
            for cfg in configs[:-1]:
                small = materialize(model64, cfg)
                with torch.no_grad():
                    want = small(input_ids=batch).logits
                    with ctrl64.use(cfg):
                        got = model64(input_ids=batch).logits
                exact[cfg.name] = (want - got).abs().max().item()
            report["float64_max_abs_diff_vs_sliced"] = exact
            del model64
        finally:
            undo()

    # 3. importance ordering
    def table():
        return {c.name: perplexity(model, ctrl, c, pairs(held))["perplexity"] for c in configs}

    report["heldout_perplexity_original_order"] = table()
    importance = collect_importance(model, arch, pairs(calib), controller=ctrl)
    perm = permutation_from_importance(importance)
    report["permutation_max_logit_diff_fp32"] = verify_permutation_equivalence(model, arch, perm, batch)
    report["heldout_perplexity_importance_ordered"] = table()
    report["note"] = (
        "Smoke check of the mechanism on a small text sample with no training. Calibration and "
        "held-out text come from the same files, so these perplexities are not experimental results."
    )

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
