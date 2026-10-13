import json

import pytest
import torch

from elastoformer_moo.config import ElasticConfig, make_grid
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.train import TrainConfig, build_teacher, nested_micro_step, train

from conftest import tiny_model


def setup(family="llama", seed=0):
    model = tiny_model(family, seed=seed)
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16)
    return model, ctrl, configs


def stream(seed):
    gen = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, 128, (2, 12), generator=gen)


def test_teacher_is_independent_of_the_students_active_config(token_batch):
    model, ctrl, configs = setup()
    with torch.no_grad():
        full = model(input_ids=token_batch).logits.clone()
    teacher = build_teacher(model, ctrl)
    ctrl.set(configs[0])
    with torch.no_grad():
        teacher_out = teacher(input_ids=token_batch).logits
    assert torch.equal(teacher_out, full)  # not affected by the student's config
    assert ctrl.cfg == configs[0]  # build_teacher restored the controller
    assert all(not p.requires_grad for p in teacher.parameters())
    with torch.no_grad(), ctrl.use(configs[0]):
        assert not torch.equal(model(input_ids=token_batch).logits, full)  # student patches still active


def test_teacher_does_not_follow_student_updates(token_batch):
    model, ctrl, _ = setup()
    teacher = build_teacher(model, ctrl)
    with torch.no_grad():
        before = teacher(input_ids=token_batch).logits.clone()
        for p in model.parameters():
            p.add_(0.1)
        assert torch.equal(teacher(input_ids=token_batch).logits, before)


def test_micro_step_produces_finite_losses_and_gradients(token_batch):
    model, ctrl, configs = setup()
    teacher = build_teacher(model, ctrl)
    model.train()
    metrics = nested_micro_step(model, ctrl, configs, token_batch, TrainConfig(), teacher)
    assert set(metrics["configs"]) == {c.name for c in configs}
    assert all(v["ce"] > 0 and v["kd"] >= 0 for v in metrics["configs"].values())
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_largest_config_updates_weights_the_smallest_does_not_touch(token_batch):
    model, ctrl, configs = setup()
    teacher = build_teacher(model, ctrl)
    model.train()
    model.zero_grad()
    nested_micro_step(model, ctrl, [configs[-1]], token_batch, TrainConfig(), teacher)
    grad = model.model.layers[0].mlp.gate_proj.weight.grad
    assert grad[configs[0].mlp_width :].abs().sum() > 0


def test_monotonicity_path_runs_and_reports_penalty(token_batch):
    model, ctrl, configs = setup()
    teacher = build_teacher(model, ctrl)
    model.train()
    metrics = nested_micro_step(model, ctrl, configs, token_batch, TrainConfig(mono_weight=0.5), teacher)
    assert metrics["mono_penalty"] >= 0
    assert all(p.grad is not None for p in model.parameters())


def test_self_full_teacher_needs_no_extra_model(token_batch):
    model, ctrl, configs = setup()
    model.train()
    metrics = nested_micro_step(model, ctrl, configs, token_batch, TrainConfig(teacher="self_full"), None)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_yaml_style_string_float_is_rejected_with_a_hint():
    # PyYAML turns `lr: 2e-5` into the string '2e-5'; that must fail loudly, not inside AdamW
    with pytest.raises(TypeError, match="2.0e-5"):
        TrainConfig(lr="2e-5").validate()
    with pytest.raises(TypeError, match="steps"):
        TrainConfig(steps=1.5).validate()
    with pytest.raises(TypeError, match="autocast_bf16"):
        TrainConfig(autocast_bf16="yes").validate()
    TrainConfig(lr=2e-5, betas=[0.9, 0.95]).validate()  # lists from YAML are fine


def test_config_validation():
    with pytest.raises(ValueError, match="teacher must be"):
        TrainConfig(teacher="bogus").validate()
    with pytest.raises(ValueError, match="requires a teacher"):
        TrainConfig(teacher="none", kd_weight=1.0).validate()
    with pytest.raises(ValueError, match="at least 1"):
        TrainConfig(steps=0).validate()


def test_train_requires_a_teacher_model_when_distilling(tmp_path):
    model, ctrl, configs = setup()
    with pytest.raises(ValueError, match="needs a teacher model"):
        train(model, ctrl, configs, stream, TrainConfig(steps=1), tmp_path, teacher=None)


def test_train_loop_logs_and_changes_weights(tmp_path):
    model, ctrl, configs = setup()
    before = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    teacher = build_teacher(model, ctrl)
    tcfg = TrainConfig(steps=4, lr=1e-3, warmup_steps=1, log_every=2, grad_accum=2)
    history = train(model, ctrl, configs, stream, tcfg, tmp_path, teacher=teacher)
    assert [h["step"] for h in history] == [2, 4]
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [2, 4]
    assert not torch.equal(before, model.model.layers[0].mlp.down_proj.weight)
    assert (tmp_path / "train_config.json").exists()


def test_resume_restores_weights_and_step(tmp_path):
    model, ctrl, configs = setup(seed=0)
    teacher = build_teacher(model, ctrl)
    tcfg = TrainConfig(steps=2, lr=1e-3, warmup_steps=1, log_every=1)
    train(model, ctrl, configs, stream, tcfg, tmp_path, teacher=teacher)
    trained = {k: v.clone() for k, v in model.state_dict().items()}

    fresh, fresh_ctrl, _ = setup(seed=99)  # different initial weights
    fresh_teacher = build_teacher(fresh, fresh_ctrl)
    resumed = train(fresh, fresh_ctrl, configs, stream, TrainConfig(steps=2, warmup_steps=1), tmp_path,
                    teacher=fresh_teacher, resume=True)
    assert resumed == []  # nothing left to run
    for name, tensor in fresh.state_dict().items():
        assert torch.equal(tensor, trained[name]), name

    longer = train(fresh, fresh_ctrl, configs, stream, TrainConfig(steps=4, lr=1e-3, warmup_steps=1, log_every=1),
                   tmp_path, teacher=fresh_teacher, resume=True)
    assert [h["step"] for h in longer] == [3, 4]


def test_no_distillation_mode_trains(tmp_path):
    model, ctrl, configs = setup()
    tcfg = TrainConfig(steps=2, kd_weight=0.0, teacher="none", warmup_steps=1, log_every=1)
    history = train(model, ctrl, configs, stream, tcfg, tmp_path)
    assert len(history) == 2
