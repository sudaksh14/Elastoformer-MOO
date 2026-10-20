import json

import pytest
import torch

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.cost import hv_widths, hypervolume_2d
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.train import TrainConfig, nested_micro_step, train, build_teacher
from elastoformer_moo.weights import ConfigWeights, gradient_dots, hv_utilities
from conftest import tiny_model


def test_hv_widths_equal_the_finite_difference_of_the_hypervolume():
    costs, losses, ref_scale = [1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.5, 1.0], 1.25
    widths = hv_widths(costs, losses, ref_scale)
    ref = (ref_scale * max(costs), 10.0)
    base = hypervolume_2d(list(zip(costs, losses)), ref)
    for i, width in enumerate(widths):
        bumped = list(losses); bumped[i] -= 1e-3
        assert (hypervolume_2d(list(zip(costs, bumped)), ref) - base) / 1e-3 == pytest.approx(width, rel=1e-6)


def test_dominated_configuration_has_zero_width_and_its_cheaper_neighbour_spans_the_gap():
    # c=3 is worse than the cheaper c=2: dominated
    widths = hv_widths([1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 3.0, 1.0], 1.25)
    assert widths[2] == 0.0 and widths[1] == 2.0 and widths[0] == 1.0 and widths[3] == 1.0


def test_hv_widths_validation():
    with pytest.raises(ValueError):
        hv_widths([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        hv_widths([1.0, 2.0], [1.0, 0.5], ref_scale=1.0)


def setup(mode="fixed", max_weight=1.0):
    model = tiny_model("llama")
    ctrl = make_elastic(model)
    arch = ctrl.arch
    configs = [ElasticConfig(f"c{g}", kv_groups=g, mlp_width=16 * g) for g in (1, 2, 4)]
    return model, ctrl, arch, configs, ConfigWeights(configs, arch, mode, max_weight)


def test_fixed_weights_boost_only_the_largest():
    _, _, _, _, w = setup("fixed", 3.0)
    assert w.as_dict() == {"c1": 1.0, "c2": 1.0, "c4": 3.0}
    with pytest.raises(ValueError):
        ConfigWeights([], None) if False else ConfigWeights(setup()[3], setup()[2], "bogus")


def test_weighted_loss_is_a_weighted_mean_and_equal_weights_match_the_default():
    model, ctrl, arch, configs, _ = setup()
    batch = torch.randint(0, 128, (2, 12))
    tcfg = TrainConfig(kd_weight=0.0, teacher="none")
    def grads(weights):
        model.zero_grad(); nested_micro_step(model, ctrl, configs, batch, tcfg, None, weights)
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    base, equal = grads(None), grads({"c1": 1.0, "c2": 1.0, "c4": 1.0})
    assert torch.allclose(base, equal)
    scaled = grads({"c1": 5.0, "c2": 5.0, "c4": 5.0})  # a common factor cancels in the weighted mean
    assert torch.allclose(base, scaled, atol=1e-6)
    heavy = grads({"c1": 1.0, "c2": 1.0, "c4": 10.0})
    assert not torch.allclose(base, heavy)


def test_update_moves_weight_toward_high_utility_and_respects_bounds():
    _, _, _, _, w = setup("hv")
    w.full_min = 0.0  # this test checks the ordering only; the floor has its own test
    for _ in range(4):
        new = w.update({"c1": 5.0, "c2": 0.0, "c4": -5.0})
    assert new["c1"] > new["c2"] > new["c4"]
    assert new["c1"] <= w.w_max and new["c4"] >= w.w_min


def test_full_model_weight_never_drops_below_its_floor():
    _, _, _, _, w = setup("hv")  # full_min defaults to 1.0, the starting weight
    for _ in range(20):
        new = w.update({"c1": 5.0, "c2": 3.0, "c4": -50.0})  # the full model is always the worst
    assert new["c4"] >= 1.0 - 1e-12
    assert new["c1"] > new["c4"]
    state = w.state(); w2 = setup("hv")[4]; w2.set_state(state)
    assert w2.as_dict() == w.as_dict()


def test_gradient_dots_are_consistent_and_leave_parameter_grads_untouched():
    model, ctrl, arch, configs, _ = setup()
    batch = torch.randint(0, 128, (2, 12)); val = [torch.randint(0, 128, (2, 12))]
    def loss_fn(m, c, cfg, b, training):
        from elastoformer_moo.losses import lm_cross_entropy
        with c.use(cfg):
            return lm_cross_entropy(m(input_ids=b).logits, b)
    dots, vals = gradient_dots(model, ctrl, configs, batch, val, loss_fn, False)
    assert len(dots) == 3 and len(dots[0]) == 3 and all(v > 0 for v in vals)
    assert all(p.grad is None for p in model.parameters())
    # same batch for train and validation: cosines, so the diagonal is 1 and the matrix is symmetric
    same, _ = gradient_dots(model, ctrl, configs, batch, [batch], loss_fn, False)
    assert all(same[i][i] == pytest.approx(1.0, abs=1e-6) for i in range(3))
    assert same[0][1] == pytest.approx(same[1][0], rel=1e-4)
    assert all(-1.0 - 1e-6 <= same[i][j] <= 1.0 + 1e-6 for i in range(3) for j in range(3))
    u = hv_utilities(same, [1.0, 2.0, 4.0], [3.0, 2.0, 1.0])
    assert len(u) == 3


def test_train_in_hv_mode_updates_weights_logs_and_resumes(tmp_path):
    model, ctrl, arch, configs, w = setup("hv")
    data = [torch.randint(0, 128, (2, 12)) for _ in range(64)]
    make_batches = lambda seed: iter(data)
    tcfg = TrainConfig(steps=6, lr=1e-3, kd_weight=0.0, teacher="none", weight_mode="hv", hv_interval=2, hv_val_batches=1,
                       log_every=2, save_every=2)
    from elastoformer_moo.cost import make_cost_fn
    cost = make_cost_fn("params", arch, configs)
    history = train(model, ctrl, configs, make_batches, tcfg, tmp_path, weights=w, hv_val_batches=lambda: [data[0]], cost_fn=cost)
    rows = [json.loads(l) for l in (tmp_path / "hv_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4]
    assert history[-1]["weights"] and history[-1]["weights"]["c4"] >= 1.0 - 1e-12
    # resume restores the weights
    w2 = ConfigWeights(configs, arch, "hv"); tcfg2 = TrainConfig(**{**tcfg.__dict__, "steps": 8})
    train(model, ctrl, configs, make_batches, tcfg2, tmp_path, weights=w2, hv_val_batches=lambda: [data[0]], cost_fn=cost, resume=True)
    assert w2.as_dict() != {"c1": 1.0, "c2": 1.0, "c4": 1.0}


def test_hv_mode_requires_validation_batches_and_cost(tmp_path):
    model, ctrl, arch, configs, w = setup("hv")
    with pytest.raises(ValueError, match="hv_val_batches"):
        train(model, ctrl, configs, lambda s: iter([]), TrainConfig(steps=1, kd_weight=0.0, teacher="none"), tmp_path, weights=w)
