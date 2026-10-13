import pytest
import torch
from transformers import AutoModelForCausalLM

from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.export import materialize
from elastoformer_moo.nested import make_elastic

from conftest import physical_reference, tiny_model

CFG = ElasticConfig("g2_m48", kv_groups=2, mlp_width=48)


def test_library_slicing_agrees_with_the_independent_test_oracle(family):
    """conftest.physical_reference is a separate implementation; the two must produce identical weights."""
    model = tiny_model(family)
    ours, oracle = materialize(model, CFG).state_dict(), physical_reference(model, CFG).state_dict()
    assert ours.keys() == oracle.keys()
    for name in ours:
        assert torch.equal(ours[name], oracle[name]), name


def test_materialized_model_has_the_active_parameter_count(family):
    model = tiny_model(family)
    small = materialize(model, CFG)
    assert sum(p.numel() for p in small.parameters()) == make_elastic(model).active_parameters(CFG)
    assert small.config.num_key_value_heads == 2 and small.config.intermediate_size == 48


def test_materialized_model_survives_save_and_load(family, tmp_path, token_batch):
    model = tiny_model(family)
    small = materialize(model, CFG)
    small.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=token_batch).logits, small(input_ids=token_batch).logits)


def test_materialize_is_a_copy_not_a_view(family):
    model = tiny_model(family)
    small = materialize(model, CFG)
    before = model.model.layers[0].mlp.gate_proj.weight.clone()
    small.model.layers[0].mlp.gate_proj.weight.data.add_(1.0)
    assert torch.equal(model.model.layers[0].mlp.gate_proj.weight, before)


def test_materialize_rejects_an_invalid_config():
    with pytest.raises(ValueError, match="kv_groups"):
        materialize(tiny_model("llama"), ElasticConfig("bad", kv_groups=9, mlp_width=48))
