import math

import numpy as np
import pytest
import torch

from elastoformer_moo.arch import SUPPORTED_MODEL_TYPES
from elastoformer_moo.config import make_grid
from elastoformer_moo.data import TokenBlocks
from elastoformer_moo.evaluate import evaluate_configs, perplexity
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.registry import get_spec, load_registry

from conftest import tiny_model


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint32).tofile(path)
    return path


def test_blocks_have_the_right_shape_and_content(token_file):
    blocks = TokenBlocks(token_file, seq_len=10)
    assert len(blocks) == 100
    assert blocks.block(3).tolist() == list(range(30, 40))
    assert blocks.block(0).dtype == torch.int64


def test_sequential_batches_cover_the_file_in_order(token_file):
    batches = list(TokenBlocks(token_file, seq_len=10).sequential(batch_size=32))
    assert [b.shape[0] for b in batches] == [32, 32, 32, 4]
    assert batches[0][0, 0].item() == 0 and batches[1][0, 0].item() == 320


def test_sequential_respects_max_batches(token_file):
    assert len(list(TokenBlocks(token_file, 10).sequential(8, max_batches=2))) == 2


def test_random_stream_is_seeded(token_file):
    blocks = TokenBlocks(token_file, seq_len=10)
    a, b, c = blocks.random(4, seed=1), blocks.random(4, seed=1), blocks.random(4, seed=2)
    first = next(a)
    assert torch.equal(first, next(b)) and not torch.equal(first, next(c))
    assert first.shape == (4, 10)
    assert all(row.tolist() == list(range(row[0].item(), row[0].item() + 10)) for row in first)  # contiguous blocks


def test_token_file_errors(tmp_path, token_file):
    with pytest.raises(FileNotFoundError):
        TokenBlocks(tmp_path / "missing.bin", 10)
    with pytest.raises(ValueError, match="need more than"):
        TokenBlocks(token_file, seq_len=5000)


def test_default_registry_loads_and_flags_support():
    registry = load_registry()
    assert registry["llama3.2-1b"].supported and registry["llama3.2-1b"].gated
    for name in ("tinyllama-1.1b", "mobilellama-1.4b", "smollm2-360m", "qwen2.5-1.5b", "qwen3-0.6b"):
        assert registry[name].supported, name
    for name in ("gemma-2b", "phi-3-mini", "smollm3-3b"):
        assert not registry[name].supported, name
    assert {s.model_type for s in registry.values() if s.supported} <= set(SUPPORTED_MODEL_TYPES)


def test_verified_flags_agree_with_the_recorded_hub_facts():
    """`verified: true` must be backed by results/e0_model_facts_*.json, never set by hand."""
    import json
    from pathlib import Path

    facts_file = sorted((Path(__file__).resolve().parents[1] / "results").glob("e0_model_facts_*.json"))[-1]
    facts = json.loads(facts_file.read_text())
    registry = load_registry()
    for name, spec in registry.items():
        row = facts[name]
        backed = row.get("model_type_matches_registry") is True and "error" not in row
        assert spec.verified == backed, f"{name}: verified={spec.verified} but facts say {backed}"
        if spec.verified:
            assert spec.license, name
    # gated or too-new models could not be checked and must not be marked verified
    for name in ("llama3.2-1b", "llama3.2-3b", "gemma-2b", "smollm3-3b"):
        assert not registry[name].verified


def test_registry_errors(tmp_path):
    with pytest.raises(KeyError, match="unknown model"):
        get_spec("nope")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  x:\n    hf_id: a/b\n")
    with pytest.raises(ValueError, match="missing fields"):
        load_registry(bad)
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "none.yaml")


def test_perplexity_is_exp_of_nll_and_full_config_matches_none():
    model = tiny_model("llama")
    ctrl = make_elastic(model)
    batches = [torch.randint(0, 128, (2, 12)) for _ in range(3)]
    full_cfg = make_grid(ctrl.arch, fractions=(1.0,))[0]
    a = perplexity(model, ctrl, None, batches)
    b = perplexity(model, ctrl, full_cfg, batches)
    assert a["perplexity"] == pytest.approx(math.exp(a["nll"]))
    assert a["perplexity"] == pytest.approx(b["perplexity"], rel=1e-6)
    assert a["tokens"] == 3 * 2 * 11


def test_evaluate_configs_reports_growing_parameter_counts():
    model = tiny_model("qwen3")
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 0.75, 1.0), mlp_multiple=16)
    batches = [torch.randint(0, 128, (2, 12)) for _ in range(2)]
    rows = evaluate_configs(model, ctrl, configs, lambda: batches)
    params = [r["active_parameters"] for r in rows]
    assert params == sorted(params) and params[-1] == sum(p.numel() for p in model.parameters())
    assert all(math.isfinite(r["perplexity"]) for r in rows)
    with pytest.raises(ValueError, match="no evaluation tokens"):
        perplexity(model, ctrl, None, [])


def test_epoch_stream_visits_every_block_once_per_epoch_and_ranks_are_disjoint(token_file):
    blocks = TokenBlocks(token_file, seq_len=10)  # 100 blocks
    seen = []
    for rank in (0, 1):
        stream = blocks.epoch_stream(batch_size=5, seed=3, rank=rank, world=2)
        starts = [row[0].item() // 10 for _ in range(10) for row in next(stream)]  # 10 batches x 5 = one epoch share
        assert len(set(starts)) == 50  # no repeats within the epoch
        seen.append(set(starts))
    assert seen[0].isdisjoint(seen[1]) and len(seen[0] | seen[1]) == 100


def test_epoch_stream_is_deterministic_and_resumes_exactly(token_file):
    blocks = TokenBlocks(token_file, seq_len=10)
    full = blocks.epoch_stream(batch_size=4, seed=9)
    first = [next(full) for _ in range(30)]  # crosses an epoch boundary (25 batches per epoch)
    resumed = blocks.epoch_stream(batch_size=4, seed=9, skip_batches=17)
    for original, again in zip(first[17:], [next(resumed) for _ in range(13)]):
        assert torch.equal(original, again)
    other_seed = next(blocks.epoch_stream(batch_size=4, seed=10))
    assert not torch.equal(first[0], other_seed)


def test_epoch_stream_rejects_too_little_data(token_file):
    with pytest.raises(ValueError, match="too few"):
        next(TokenBlocks(token_file, seq_len=10).epoch_stream(batch_size=60, seed=0, world=2))


def test_batched_tokenization_matches_one_at_a_time(tmp_path):
    class Tok:
        eos_token_id = 1
        calls = 0

        def __call__(self, text, add_special_tokens=False):
            Tok.calls += 1
            many = isinstance(text, list)
            ids = [[ord(c) % 40 + 3 for c in t] for t in (text if many else [text])]
            return {"input_ids": ids if many else ids[0]}

    from elastoformer_moo.data import write_token_file

    docs = ["alpha", "bb", "gamma delta", "e", "zeta"]
    a = write_token_file(docs, Tok(), tmp_path / "a.bin")
    b = write_token_file(docs, Tok(), tmp_path / "b.bin", batch_size=2)
    assert a == b and (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    assert write_token_file(docs, Tok(), tmp_path / "c.bin", max_tokens=9, batch_size=3) == 9


def test_perplexity_under_bf16_autocast_is_finite_and_close_to_fp32():
    model = tiny_model("llama")
    ctrl = make_elastic(model)
    batches = [torch.randint(0, 128, (2, 12)) for _ in range(3)]
    plain = perplexity(model, ctrl, None, batches)
    autocast = perplexity(model, ctrl, None, batches, autocast_bf16=True)
    assert math.isfinite(autocast["perplexity"])
    assert autocast["perplexity"] == pytest.approx(plain["perplexity"], rel=5e-2)  # bf16 noise, not a different answer


def test_evaluate_configs_passes_autocast_through():
    model = tiny_model("qwen2")
    ctrl = make_elastic(model)
    configs = make_grid(ctrl.arch, fractions=(0.5, 1.0), mlp_multiple=16)
    batches = [torch.randint(0, 128, (2, 12))]
    rows = evaluate_configs(model, ctrl, configs, lambda: batches, autocast_bf16=True)
    assert len(rows) == 2 and all(math.isfinite(r["perplexity"]) for r in rows)
