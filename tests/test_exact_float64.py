"""Exactness tests in float64.

Hugging Face RMSNorm casts to float32 internally, which would put float32 rounding noise into
a float64 run and hide whether two paths are mathematically identical. These tests patch it to
stay in the input dtype, so the comparison can demand ~1e-12 instead of a loose tolerance.
A difference at that level means the elastic forward is the same function as a physically
sliced model, not merely close to it.
"""

import importlib

import pytest
import torch

import elastoformer_moo.permute as permute_module
from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.importance import Importance
from elastoformer_moo.nested import make_elastic
from elastoformer_moo.permute import Permutation, permutation_from_importance, verify_permutation_equivalence

from conftest import physical_reference, tiny_model

CONFIGS = [
    ElasticConfig("g2_m48", kv_groups=2, mlp_width=48),
    ElasticConfig("g1_m32", kv_groups=1, mlp_width=32),
    ElasticConfig("g3_m96", kv_groups=3, mlp_width=96),
    ElasticConfig("g4_m16", kv_groups=4, mlp_width=16),
]


@pytest.fixture
def native_dtype_norm(family, monkeypatch):
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    norm_cls = next(getattr(module, name) for name in dir(module) if name.endswith("RMSNorm"))

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    monkeypatch.setattr(norm_cls, "forward", forward)


def double_model(family):
    return tiny_model(family).double()


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_elastic_forward_equals_the_sliced_model_to_machine_precision(family, cfg, token_batch, native_dtype_norm):
    model = double_model(family)
    reference = physical_reference(model, cfg).double().eval()
    ctrl = make_elastic(model)
    with torch.no_grad():
        expected = reference(input_ids=token_batch).logits
        with ctrl.use(cfg):
            got = model(input_ids=token_batch).logits
    assert got.dtype == torch.float64
    assert (expected - got).abs().max().item() <= 1e-12


def random_permutation(arch, seed):
    g = torch.Generator().manual_seed(seed)
    imp = Importance(
        mlp=[torch.rand(arch.intermediate_size, generator=g) for _ in range(arch.num_layers)],
        groups=[torch.rand(arch.num_kv_heads, generator=g) for _ in range(arch.num_layers)],
    )
    return permutation_from_importance(imp)


def test_permutation_preserves_the_function_to_machine_precision(family, token_batch, native_dtype_norm):
    model = double_model(family)
    arch = ArchInfo.from_hf_config(model.config)
    diff = verify_permutation_equivalence(model, arch, random_permutation(arch, 7), token_batch, atol=1e-10, rtol=0.0)
    assert diff <= 1e-10


def test_permuted_model_still_matches_the_oracle_exactly(family, token_batch, native_dtype_norm):
    """Permute, then slice: the prefix of the permuted model must equal a physically sliced permuted model."""
    model = double_model(family)
    arch = ArchInfo.from_hf_config(model.config)
    verify_permutation_equivalence(model, arch, random_permutation(arch, 3), token_batch, atol=1e-10, rtol=0.0)
    cfg = CONFIGS[0]
    reference = physical_reference(model, cfg).double().eval()
    ctrl = make_elastic(model)
    with torch.no_grad():
        expected = reference(input_ids=token_batch).logits
        with ctrl.use(cfg):
            got = model(input_ids=token_batch).logits
    assert (expected - got).abs().max().item() <= 1e-12


def test_verification_catches_an_inconsistent_permutation(monkeypatch, token_batch):
    """The relative tolerance must not be loose enough to accept a half-applied permutation."""
    model = tiny_model("llama")
    arch = ArchInfo.from_hf_config(model.config)
    perm = random_permutation(arch, 5)
    real_apply = permute_module.apply_permutation

    def half_applied(m, a, p):
        real_apply(m, a, p)  # permutes gate/up AND down ...
        for layer, order in zip(m.model.layers, p.mlp):  # ... then undoes only down_proj
            inverse = torch.argsort(order)
            layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, inverse].clone()

    monkeypatch.setattr(permute_module, "apply_permutation", half_applied)
    with pytest.raises(AssertionError, match="changed the logits"):
        verify_permutation_equivalence(model, arch, perm, token_batch)  # default atol and rtol
