import json

import numpy as np
import pytest
import torch

from elastoformer_moo.data import write_token_file
from elastoformer_moo.pipeline import (
    apply_overrides,
    evaluate_pretrained,
    load_causal_lm,
    permute_pretrained,
    resolve_model_path,
    run_training,
)

from conftest import tiny_model


@pytest.fixture
def tiny_checkpoint(tmp_path):
    path = tmp_path / "tiny"
    tiny_model("llama").save_pretrained(path)
    return path


@pytest.fixture
def token_files(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("train", "val", "calib"):
        path = tmp_path / f"{name}.bin"
        rng.integers(0, 128, size=3000).astype(np.uint32).tofile(path)
        paths[name] = path
    return paths


def test_permute_pretrained_preserves_logits_and_writes_artifacts(tiny_checkpoint, token_files, tmp_path):
    out = tmp_path / "permuted"
    report = permute_pretrained(
        str(tiny_checkpoint), out, token_files["calib"], seq_len=16, batch_size=2, n_batches=4,
        save_dtype="float32",
    )
    assert report["max_logit_diff"] <= report["atol"]
    for name in ("config.json", "permutation.pt", "importance.pt", "permutation_report.json"):
        assert (out / name).exists(), name
    original, permuted = load_causal_lm(str(tiny_checkpoint)), load_causal_lm(str(out))
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(original(input_ids=ids).logits, permuted(input_ids=ids).logits, atol=1e-4, rtol=1e-4)


def test_evaluate_pretrained_reports_each_configuration(tiny_checkpoint, token_files):
    rows = evaluate_pretrained(
        str(tiny_checkpoint), token_files["val"], seq_len=16, batch_size=2,
        fractions=(0.5, 1.0), mlp_multiple=16, max_batches=2,
    )
    assert [r["config"] for r in rows] == ["f050", "f100"]
    assert rows[0]["active_parameters"] < rows[1]["active_parameters"]


def test_run_training_end_to_end_on_cpu(tiny_checkpoint, token_files, tmp_path):
    cfg = {
        "model": str(tiny_checkpoint),
        "device": "cpu",
        "out_dir": str(tmp_path / "run"),
        "elastic": {"fractions": [0.5, 0.75, 1.0], "mlp_multiple": 16},
        "train": {"steps": 3, "lr": 1e-3, "warmup_steps": 1, "log_every": 1, "grad_clip": 1.0},
        "data": {
            "train_tokens": str(token_files["train"]), "val_tokens": str(token_files["val"]),
            "seq_len": 16, "batch_size": 2, "eval_batches": 2,
        },
    }
    result = run_training(cfg)
    out = tmp_path / "run"
    assert result["final_step"] == 3
    assert len(result["eval"]) == 3
    assert (out / "final" / "config.json").exists()
    assert len(json.loads((out / "configs.json").read_text())) == 3
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 3
    # the saved checkpoint loads back as an ordinary Hugging Face model
    assert load_causal_lm(str(out / "final")).config.num_key_value_heads == 4


def test_apply_overrides_parses_yaml_values():
    cfg = {"train": {"steps": 10}, "data": {}}
    apply_overrides(cfg, ["train.steps=50", "train.lr=2e-5", "data.new.flag=true", "name=abc", "x=-1.5E+3", "y=inf"])
    assert cfg["x"] == -1500.0 and cfg["y"] == "inf"  # only scientific notation is rescued
    assert cfg["train"]["steps"] == 50
    assert cfg["train"]["lr"] == pytest.approx(2e-5)
    assert cfg["data"]["new"]["flag"] is True
    assert cfg["name"] == "abc"
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["oops"])
    with pytest.raises(ValueError, match="cannot descend"):
        apply_overrides(cfg, ["name.sub=1"])


def test_resolve_model_path(tmp_path):
    assert resolve_model_path(str(tmp_path)) == str(tmp_path)  # local directory wins
    assert resolve_model_path("llama3.2-1b") == "meta-llama/Llama-3.2-1B"  # registry name
    assert resolve_model_path("some/hub-id") == "some/hub-id"  # passed through


def test_load_causal_lm_rejects_bad_dtype(tiny_checkpoint):
    with pytest.raises(ValueError, match="dtype"):
        load_causal_lm(str(tiny_checkpoint), dtype="int4")


def test_write_token_file_appends_eos(tmp_path):
    class FakeTokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 50 + 3 for c in text]}

    path = tmp_path / "out" / "tokens.bin"
    total = write_token_file(["abc", "de"], FakeTokenizer(), path)
    tokens = np.fromfile(path, dtype=np.uint32)
    assert total == 7 and len(tokens) == 7
    assert tokens[3] == 2 and tokens[-1] == 2
    assert write_token_file(["ab"], FakeTokenizer(), path, append_eos=False) == 2
