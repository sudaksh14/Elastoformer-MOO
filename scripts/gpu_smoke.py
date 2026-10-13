"""GPU smoke test: CUDA is visible, elastic forward matches the CPU result, and one nested
training step runs on the GPU. Exits non-zero on any failure; exits 2 if CUDA is unavailable.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from elastoformer_moo.config import make_grid  # noqa: E402
from elastoformer_moo.nested import make_elastic  # noqa: E402
from elastoformer_moo.train import TrainConfig, build_teacher, nested_micro_step  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available; this smoke test must run on a GPU node.", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    torch.manual_seed(0)
    cpu_model = LlamaForCausalLM(
        LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                    num_attention_heads=8, num_key_value_heads=4, head_dim=8, max_position_embeddings=64)
    ).eval()
    gpu_model = copy.deepcopy(cpu_model).to(device)  # copy BEFORE attaching: patched closures must not be shared
    cpu_ctrl, gpu_ctrl = make_elastic(cpu_model), make_elastic(gpu_model)
    configs = make_grid(cpu_ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16)

    ids = torch.randint(0, 128, (2, 12))
    worst = 0.0
    for cfg in configs:
        with torch.no_grad(), cpu_ctrl.use(cfg):
            expected = cpu_model(input_ids=ids).logits
        with torch.no_grad(), gpu_ctrl.use(cfg):
            got = gpu_model(input_ids=ids.to(device)).logits.cpu()
        worst = max(worst, (expected - got).abs().max().item())
    if worst > 1e-3:
        print(f"GPU and CPU logits differ by {worst:.3e}", file=sys.stderr)
        return 1

    gpu_model.train()
    teacher = build_teacher(gpu_model, gpu_ctrl)
    tcfg = TrainConfig(autocast_bf16=True)
    metrics = nested_micro_step(gpu_model, gpu_ctrl, configs, ids.to(device), tcfg, teacher)
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu_model.parameters()):
        print("non-finite or missing gradients after the nested step", file=sys.stderr)
        return 1
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "max_cpu_gpu_logit_diff": worst,
        "nested_step_loss": metrics["loss"],
        "status": "ok",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
