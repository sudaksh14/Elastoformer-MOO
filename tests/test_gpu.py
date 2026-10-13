"""GPU tests. Skipped when CUDA is unavailable; run them on DAS-6 nodes 205-208 via run_gpu_tests.sh."""

import copy

import pytest
import torch

from elastoformer_moo.config import make_grid
from elastoformer_moo.nested import make_elastic

from conftest import tiny_model

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")]


def _flash_importable() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _flash_importable(), reason="flash_attn cannot be imported here")
@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "mistral"])
def test_flash_attention_agrees_with_sdpa_for_every_config(family):
    """Flash and sdpa are different kernels over the same weights: they must agree within bf16 noise,
    and the elastic forward must still equal a materialized standalone model under flash."""
    from elastoformer_moo.export import materialize

    base = tiny_model(family).to(torch.bfloat16)
    sdpa_model = copy.deepcopy(base).cuda()  # copy before attaching the elastic patches
    flash_model = copy.deepcopy(base).cuda()
    sdpa_model.config._attn_implementation = "sdpa"
    flash_model.config._attn_implementation = "flash_attention_2"
    sdpa_ctrl, flash_ctrl = make_elastic(sdpa_model), make_elastic(flash_model)
    ids = torch.randint(0, 128, (2, 32)).cuda()
    for cfg in make_grid(flash_ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16):
        with torch.no_grad(), sdpa_ctrl.use(cfg):
            expected = sdpa_model(input_ids=ids).logits.float()
        with torch.no_grad(), flash_ctrl.use(cfg):
            got = flash_model(input_ids=ids).logits.float()
        torch.testing.assert_close(got, expected, atol=5e-2, rtol=5e-2)
        standalone = materialize(flash_model, cfg).cuda().eval()
        with torch.no_grad():
            torch.testing.assert_close(standalone(input_ids=ids).logits.float(), got, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "mistral"])
def test_gpu_logits_match_cpu_for_every_config(family):
    torch.backends.cuda.matmul.allow_tf32 = False
    cpu_model = tiny_model(family)
    gpu_model = copy.deepcopy(cpu_model).cuda()  # copy before attaching the elastic patches
    cpu_ctrl, gpu_ctrl = make_elastic(cpu_model), make_elastic(gpu_model)
    ids = torch.randint(0, 128, (2, 12))
    for cfg in make_grid(cpu_ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16):
        with torch.no_grad(), cpu_ctrl.use(cfg):
            expected = cpu_model(input_ids=ids).logits
        with torch.no_grad(), gpu_ctrl.use(cfg):
            got = gpu_model(input_ids=ids.cuda()).logits.cpu()
        torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)
