import pytest
import torch

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.nested import make_elastic

from conftest import physical_reference, tiny_model

CONFIGS = [
    ElasticConfig("g2_m48", kv_groups=2, mlp_width=48),
    ElasticConfig("g1_m32", kv_groups=1, mlp_width=32),
    ElasticConfig("g3_m96", kv_groups=3, mlp_width=96),  # MLP full, attention reduced
    ElasticConfig("g4_m16", kv_groups=4, mlp_width=16),  # attention full, MLP reduced
]


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_full_config_matches_original_exactly(family, token_batch):
    model = tiny_model(family)
    reference = logits(model, token_batch)
    ctrl = make_elastic(model)
    assert torch.equal(logits(model, token_batch), reference)  # no config active
    full = ElasticConfig("full", kv_groups=4, mlp_width=96)
    with ctrl.use(full):
        assert torch.equal(logits(model, token_batch), reference)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_elastic_forward_matches_physically_sliced_model(family, cfg, token_batch):
    model = tiny_model(family)
    small = physical_reference(model, cfg)
    expected = logits(small, token_batch)
    ctrl = make_elastic(model)
    with ctrl.use(cfg):
        got = logits(model, token_batch)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS[:2], ids=lambda c: c.name)
def test_active_parameter_count_matches_physical_model(family, cfg):
    model = tiny_model(family)
    small = physical_reference(model, cfg)
    ctrl = make_elastic(model)
    assert ctrl.active_parameters(cfg) == sum(p.numel() for p in small.parameters())
    assert ctrl.active_parameters(None) == sum(p.numel() for p in model.parameters())


def test_state_dict_keys_unchanged_by_attach(family):
    model = tiny_model(family)
    before = list(model.state_dict().keys())
    ctrl = make_elastic(model)
    assert list(model.state_dict().keys()) == before
    ctrl.detach()
    assert list(model.state_dict().keys()) == before


def test_detach_restores_original_behaviour(family, token_batch):
    model = tiny_model(family)
    reference = logits(model, token_batch)
    ctrl = make_elastic(model)
    ctrl.set(CONFIGS[0])
    ctrl.detach()
    assert torch.equal(logits(model, token_batch), reference)


def test_use_restores_previous_config(family):
    model = tiny_model(family)
    ctrl = make_elastic(model)
    ctrl.set(CONFIGS[0])
    with ctrl.use(CONFIGS[1]):
        assert ctrl.cfg == CONFIGS[1]
    assert ctrl.cfg == CONFIGS[0]


def test_invalid_config_rejected_on_set(family):
    ctrl = make_elastic(tiny_model(family))
    with pytest.raises(ValueError, match="kv_groups"):
        ctrl.set(ElasticConfig("bad", kv_groups=5, mlp_width=48))


def test_smaller_config_changes_output(family, token_batch):
    model = tiny_model(family)
    ctrl = make_elastic(model)
    full = logits(model, token_batch)
    with ctrl.use(CONFIGS[1]):
        small = logits(model, token_batch)
    assert not torch.allclose(full, small)


def test_greedy_generation_with_kv_cache_matches_physical_model(family):
    model = tiny_model(family)
    cfg = CONFIGS[0]
    small = physical_reference(model, cfg)
    ctrl = make_elastic(model)
    prompt = torch.tensor([[5, 17, 42, 8]])
    expected = small.generate(prompt, max_new_tokens=6, do_sample=False, pad_token_id=0)
    with ctrl.use(cfg):
        got = model.generate(prompt, max_new_tokens=6, do_sample=False, pad_token_id=0)
    assert torch.equal(got, expected)


def test_gradients_reach_shared_weights_outside_the_smallest_config(family, token_batch):
    """Training the smallest config updates only its prefix; the largest touches the rest."""
    model = tiny_model(family)
    ctrl = make_elastic(model)
    small = CONFIGS[1]  # kv_groups=1, mlp_width=32
    model.zero_grad()
    with ctrl.use(small):
        model(input_ids=token_batch).logits.sum().backward()
    layer = model.model.layers[0]
    assert layer.mlp.gate_proj.weight.grad[:32].abs().sum() > 0
    assert layer.mlp.gate_proj.weight.grad[32:].abs().sum() == 0
    assert layer.self_attn.k_proj.weight.grad[:8].abs().sum() > 0
    assert layer.self_attn.k_proj.weight.grad[8:].abs().sum() == 0
