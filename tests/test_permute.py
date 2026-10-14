import pytest
import torch

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.importance import Importance, collect_importance
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.permute import (
    Permutation,
    apply_permutation,
    permutation_from_importance,
    verify_permutation_equivalence,
)

from conftest import tiny_model


def random_importance(arch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Importance(
        mlp=[torch.rand(arch.intermediate_size, generator=g) for _ in range(arch.num_layers)],
        groups=[torch.rand(arch.num_kv_heads, generator=g) for _ in range(arch.num_layers)],
    )


def test_permutation_preserves_the_function(family, token_batch):
    model = tiny_model(family)
    arch = ArchInfo.from_hf_config(model.config)
    perm = permutation_from_importance(random_importance(arch))
    diff = verify_permutation_equivalence(model, arch, perm, token_batch, atol=1e-5)
    assert diff <= 1e-5


def test_most_important_structures_come_first():
    model = tiny_model("llama")
    arch = ArchInfo.from_hf_config(model.config)
    layer = model.model.layers[0]
    old_gate = layer.mlp.gate_proj.weight.detach().clone()
    old_k = layer.self_attn.k_proj.weight.detach().clone()

    mlp_scores = torch.zeros(arch.intermediate_size)
    mlp_scores[[90, 7, 33]] = torch.tensor([3.0, 2.0, 1.0])
    group_scores = torch.tensor([0.1, 0.2, 0.9, 0.3])
    imp = Importance(
        mlp=[mlp_scores.clone() for _ in range(arch.num_layers)],
        groups=[group_scores.clone() for _ in range(arch.num_layers)],
    )
    apply_permutation(model, arch, permutation_from_importance(imp))

    assert torch.equal(layer.mlp.gate_proj.weight[0], old_gate[90])
    assert torch.equal(layer.mlp.gate_proj.weight[1], old_gate[7])
    assert torch.equal(layer.mlp.gate_proj.weight[2], old_gate[33])
    # group 2 was the most important KV group, so its rows now lead
    assert torch.equal(layer.self_attn.k_proj.weight[: arch.head_dim], old_k[2 * arch.head_dim : 3 * arch.head_dim])


def test_failed_check_restores_the_original_weights(token_batch):
    model = tiny_model("qwen2")
    arch = ArchInfo.from_hf_config(model.config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    perm = permutation_from_importance(random_importance(arch, seed=3))
    with pytest.raises(AssertionError, match="changed the logits"):
        verify_permutation_equivalence(model, arch, perm, token_batch, atol=-1.0, rtol=0.0)  # always fails
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_permutation_size_mismatch_is_rejected():
    model = tiny_model("llama")
    arch = ArchInfo.from_hf_config(model.config)
    bad = Permutation(mlp=[torch.arange(10)] * arch.num_layers, groups=[torch.arange(4)] * arch.num_layers)
    with pytest.raises(ValueError, match="does not match"):
        apply_permutation(model, arch, bad)


def test_permutation_roundtrip_file(tmp_path):
    model = tiny_model("llama")
    arch = ArchInfo.from_hf_config(model.config)
    perm = permutation_from_importance(random_importance(arch))
    perm.save(str(tmp_path / "perm.pt"))
    loaded = Permutation.load(str(tmp_path / "perm.pt"))
    assert all(torch.equal(a, b) for a, b in zip(perm.mlp, loaded.mlp))


def test_collect_importance_shapes_and_hook_cleanup(family):
    model = tiny_model(family)
    arch = ArchInfo.from_hf_config(model.config)
    batches = [torch.randint(0, 128, (2, 10)) for _ in range(3)]
    imp = collect_importance(model, arch, batches)
    assert len(imp.mlp) == arch.num_layers
    assert imp.mlp[0].shape == (arch.intermediate_size,)
    assert imp.groups[0].shape == (arch.num_kv_heads,)
    assert all((s >= 0).all() for s in imp.mlp + imp.groups)
    for layer in model.model.layers:
        assert not layer.mlp.down_proj._forward_pre_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


def test_collect_importance_scores_the_full_model_even_if_a_config_is_active():
    model = tiny_model("llama")
    ctrl = make_elastic(model)
    ctrl.set(ElasticConfig("small", kv_groups=1, mlp_width=32))
    arch = ctrl.arch
    imp = collect_importance(model, arch, [torch.randint(0, 128, (2, 10))], controller=ctrl)
    assert imp.mlp[0].shape == (arch.intermediate_size,)
    assert ctrl.cfg is not None  # restored afterwards


def test_collect_importance_requires_tokens():
    model = tiny_model("llama")
    with pytest.raises(ValueError, match="no calibration tokens"):
        collect_importance(model, ArchInfo.from_hf_config(model.config), [])


@pytest.mark.parametrize("skip_first_token", [False, True])
def test_taylor_importance_shapes_no_parameter_grads_and_hook_cleanup(family, skip_first_token):
    model = tiny_model(family)
    arch = ArchInfo.from_hf_config(model.config)
    batches = [torch.randint(0, 128, (2, 10)) for _ in range(3)]
    imp = collect_importance(model, arch, batches, method="taylor", skip_first_token=skip_first_token)
    assert imp.mlp[0].shape == (arch.intermediate_size,) and imp.groups[0].shape == (arch.num_kv_heads,)
    assert all((s >= 0).all() and torch.isfinite(s).all() for s in imp.mlp + imp.groups)
    assert any(s.sum() > 0 for s in imp.mlp)
    assert all(p.grad is None for p in model.parameters())
    for layer in model.model.layers:
        assert not layer.mlp.down_proj._forward_pre_hooks and not layer.self_attn.o_proj._forward_pre_hooks


def test_skip_first_token_changes_scores_and_unknown_method_is_rejected():
    model = tiny_model("llama")
    arch = ArchInfo.from_hf_config(model.config)
    batches = [torch.randint(0, 128, (2, 10)) for _ in range(3)]
    a = collect_importance(model, arch, batches)
    b = collect_importance(model, arch, batches, skip_first_token=True)
    assert not torch.equal(a.mlp[0], b.mlp[0])
    with pytest.raises(ValueError, match="method"):
        collect_importance(model, arch, batches, method="magic")
