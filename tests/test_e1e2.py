import gzip
import io
import json
import math

import numpy as np
import pytest
import torch

from elastoformer_moo.data import write_token_file
from elastoformer_moo.e1e2 import environment_info, format_summary, run_e1_e2
from elastoformer_moo.pipeline import ATTN_IMPLEMENTATIONS, load_causal_lm
from elastoformer_moo.textdata import iter_jsonl_gz_texts, wikitext_join

from conftest import tiny_model


@pytest.fixture
def checkpoint_and_tokens(tmp_path):
    ckpt = tmp_path / "tiny"
    tiny_model("llama").save_pretrained(ckpt)
    rng = np.random.default_rng(0)
    files = {}
    for name in ("calib", "wiki", "web"):
        path = tmp_path / f"{name}.bin"
        rng.integers(0, 128, size=4000).astype(np.uint32).tofile(path)
        files[name] = path
    return ckpt, files


def test_e1_e2_end_to_end_on_cpu(checkpoint_and_tokens, tmp_path):
    ckpt, files = checkpoint_and_tokens
    out = tmp_path / "run"
    results = run_e1_e2(
        str(ckpt), out, files["calib"], {"wiki": files["wiki"], "web": files["web"]},
        seq_len=16, batch_size=2, calib_batches=4, fractions=(0.5, 1.0), mlp_multiple=16,
        dtype="float32", attn_implementation="sdpa", device="cpu", max_eval_batches=3,
    )
    assert set(results["naive_prefix"]) == set(results["importance_ordered"]) == {"wiki", "web"}
    for name in ("wiki", "web"):
        naive, ordered = results["naive_prefix"][name], results["importance_ordered"][name]
        assert [r["config"] for r in naive] == [r["config"] for r in ordered] == ["f050", "f100"]
        assert naive[-1]["ppl_ratio_to_full"] == pytest.approx(1.0)
        # E2 at the perplexity level: permuting must not change the full model's quality
        assert naive[-1]["perplexity"] == pytest.approx(ordered[-1]["perplexity"], rel=1e-3)
        assert all(math.isfinite(r["perplexity"]) for r in naive + ordered)
    assert results["permutation"]["max_logit_diff"] <= 1e-3
    assert (out / "permuted" / "permutation.pt").exists()
    assert json.loads((out / "results.json").read_text())["model"] == str(ckpt)
    summary = (out / "summary.md").read_text()
    assert "No training" in summary and "## wiki" in summary and "importance-ordered" in summary


def test_summary_names_the_environment():
    info = environment_info()
    assert {"python", "torch", "transformers", "cuda_available", "flash_attn"} <= set(info)


def test_flash_attention_is_never_a_silent_fallback(checkpoint_and_tokens):
    ckpt, _ = checkpoint_and_tokens
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        load_causal_lm(str(ckpt), dtype="float32", attn_implementation="flash_attention_2")
    with pytest.raises(ValueError, match="attn_implementation must be one of"):
        load_causal_lm(str(ckpt), attn_implementation="flash")
    assert ATTN_IMPLEMENTATIONS == ("eager", "sdpa", "flash_attention_2")


def test_flash_attention_with_float32_weights_is_allowed_only_for_autocast_training(checkpoint_and_tokens, monkeypatch):
    """Training keeps float32 master weights under bf16 autocast, so that combination must pass validation."""
    ckpt, _ = checkpoint_and_tokens
    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(kwargs)
        return tiny_model("llama")

    import transformers

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))
    load_causal_lm(str(ckpt), dtype="float32", attn_implementation="flash_attention_2", autocast=True)
    assert seen["attn_implementation"] == "flash_attention_2"


def test_write_token_file_max_tokens_stops_reading(tmp_path):
    class Tok:
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [2] * len(text)}

    consumed = []

    def texts():
        for t in ["aaaa", "bbbb", "cccc", "dddd"]:
            consumed.append(t)
            yield t

    total = write_token_file(texts(), Tok(), tmp_path / "t.bin", max_tokens=7)
    assert total == 7 and len(np.fromfile(tmp_path / "t.bin", dtype=np.uint32)) == 7
    assert consumed == ["aaaa", "bbbb"]  # the generator was not drained


def test_wikitext_join_matches_the_standard_protocol():
    assert wikitext_join(["a", "", "b"]) == "a\n\n\n\nb"


def test_jsonl_gz_streaming_reads_only_as_many_documents_as_asked():
    raw = "\n".join(json.dumps({"text": f"doc{i}"}) for i in range(10)).encode()
    buf = io.BytesIO(gzip.compress(raw))
    assert list(iter_jsonl_gz_texts(buf, max_docs=3)) == ["doc0", "doc1", "doc2"]
    assert len(list(iter_jsonl_gz_texts(io.BytesIO(gzip.compress(raw))))) == 10
