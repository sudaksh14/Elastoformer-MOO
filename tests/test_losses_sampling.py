import pytest
import torch

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.losses import distillation_kl, lm_cross_entropy, monotonicity_penalty
from elastoformer_moo.sampling import SandwichSampler

from conftest import tiny_model


def test_kl_is_zero_for_identical_logits_and_positive_otherwise():
    torch.manual_seed(0)
    a = torch.randn(2, 6, 11)
    assert distillation_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)
    assert distillation_kl(a, a + torch.randn_like(a)).item() > 0


def test_kl_matches_reference_formula():
    torch.manual_seed(0)
    s, t = torch.randn(1, 4, 7), torch.randn(1, 4, 7)
    p, q = torch.softmax(t[:, :-1], -1), torch.softmax(s[:, :-1], -1)
    expected = (p * (p.log() - q.log())).sum(-1).mean()
    assert distillation_kl(s, t).item() == pytest.approx(expected.item(), rel=1e-5)


def test_temperature_scaling_and_validation():
    torch.manual_seed(0)
    s, t = torch.randn(1, 4, 7), torch.randn(1, 4, 7)
    assert distillation_kl(s, t, temperature=2.0).item() != distillation_kl(s, t, temperature=1.0).item()
    with pytest.raises(ValueError, match="temperature"):
        distillation_kl(s, t, temperature=0)
    with pytest.raises(ValueError, match="shapes differ"):
        distillation_kl(s, torch.randn(1, 5, 7))


def test_cross_entropy_uses_next_token_targets():
    ids = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 5), -50.0)
    logits[0, 0, 2] = 50.0  # position 0 predicts token 2
    logits[0, 1, 3] = 50.0  # position 1 predicts token 3
    assert lm_cross_entropy(logits, ids).item() == pytest.approx(0.0, abs=1e-4)


def test_monotonicity_penalty_only_when_larger_is_worse():
    good = monotonicity_penalty([torch.tensor(3.0), torch.tensor(2.0), torch.tensor(1.5)], [1, 2, 3])
    assert good.item() == 0.0
    bad = monotonicity_penalty([torch.tensor(2.0), torch.tensor(2.5), torch.tensor(1.0)], [1, 2, 3])
    assert bad.item() == pytest.approx(0.5)


def test_monotonicity_penalty_orders_by_size_not_by_position():
    penalty = monotonicity_penalty([torch.tensor(1.0), torch.tensor(3.0)], [10, 1])  # size-1 config has the higher loss
    assert penalty.item() == 0.0


def test_monotonicity_penalty_edge_cases():
    assert monotonicity_penalty([torch.tensor(1.0)], [1]).item() == 0.0
    with pytest.raises(ValueError):
        monotonicity_penalty([torch.tensor(1.0)], [1, 2])


def test_penalty_gradient_flows_to_the_larger_config():
    small, large = torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
    monotonicity_penalty([small, large], [1, 2]).backward()
    assert large.grad.item() == 1.0 and small.grad.item() == -1.0


def make_sampler(n_random=1, seed=0, fractions=(2, 4, 6, 8)):
    arch = ArchInfo.from_hf_config(tiny_model("llama").config)
    configs = [ElasticConfig(f"c{g}", kv_groups=min(g, 4), mlp_width=16 * g) for g in fractions]
    return arch, configs, SandwichSampler(configs, arch, n_random=n_random, seed=seed)


def test_sandwich_always_contains_both_ends():
    _, configs, sampler = make_sampler()
    for _ in range(20):
        picked = sampler.sample()
        assert sampler.largest in picked and sampler.smallest in picked
        assert len(picked) == len(set(c.name for c in picked))


def test_sandwich_is_deterministic_for_a_seed_and_restorable():
    _, _, a = make_sampler(seed=5)
    _, _, b = make_sampler(seed=5)
    assert [x.name for _ in range(8) for x in a.sample()] == [x.name for _ in range(8) for x in b.sample()]
    state = a.state()
    first = [c.name for c in a.sample()]
    a.set_state(state)
    assert [c.name for c in a.sample()] == first


def test_sandwich_small_sets():
    _, configs, one = make_sampler(fractions=(2,))
    assert one.sample() == [configs[0]]
    _, _, two = make_sampler(fractions=(2, 4))
    assert len(two.sample()) == 2
    _, _, zero_random = make_sampler(n_random=0)
    assert len(zero_random.sample()) == 2
    with pytest.raises(ValueError):
        SandwichSampler([], ArchInfo.from_hf_config(tiny_model("llama").config))


def test_naive_sampling_draws_n_random_configs_without_forced_ends():
    arch, configs, _ = make_sampler()
    naive = SandwichSampler(configs, arch, n_random=1, seed=0, sandwich=False)
    drawn = [naive.sample() for _ in range(200)]
    assert all(len(p) == 1 for p in drawn)
    assert {p[0].name for p in drawn} == {c.name for c in configs}  # all four, ends included, by chance
    two = SandwichSampler(configs, arch, n_random=2, seed=0, sandwich=False)
    assert all(len(set(c.name for c in two.sample())) == 2 for _ in range(20))
