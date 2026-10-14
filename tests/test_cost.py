import json

import pytest

from elastoformer_moo.arch import ArchInfo
from elastoformer_moo.config import ElasticConfig
from elastoformer_moo.cost import hypervolume_2d, make_cost_fn
from elastoformer_moo.train import TrainConfig
from conftest import tiny_model


def setup():
    arch = ArchInfo.from_hf_config(tiny_model("llama").config)
    configs = [ElasticConfig(f"c{g}", kv_groups=g, mlp_width=16 * g) for g in (1, 2, 4)]
    return arch, configs


@pytest.mark.parametrize("spec", ["params", "flops", "kv_cache"])
def test_builtin_axes_increase_with_the_configuration(spec):
    arch, configs = setup()
    cost = make_cost_fn(spec, arch, configs)
    values = [cost(c) for c in configs]
    assert values == sorted(values) and len(set(values)) == 3


def test_params_axis_matches_config_size_and_flops_grow_with_sequence_length():
    arch, configs = setup()
    assert make_cost_fn("params", arch, configs)(configs[0]) == configs[0].size(arch)
    assert make_cost_fn("flops", arch, configs, 2048)(configs[0]) > make_cost_fn("flops", arch, configs, 128)(configs[0])


def test_kv_cache_bytes_per_token():
    arch, configs = setup()
    assert make_cost_fn("kv_cache", arch, configs)(configs[2]) == 2 * arch.num_layers * 4 * arch.head_dim * 2


def test_table_axis_uses_measured_values_and_must_cover_every_config(tmp_path):
    arch, configs = setup()
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({"c1": 0.5, "c2": 0.4, "c4": 0.9}))  # not monotone on purpose
    cost = make_cost_fn(f"table:{path}", arch, configs)
    assert [cost(c) for c in configs] == [0.5, 0.4, 0.9]
    path.write_text(json.dumps({"c1": 0.5}))
    with pytest.raises(ValueError, match="no entry"):
        make_cost_fn(f"table:{path}", arch, configs)


def test_unknown_axis_is_rejected():
    arch, configs = setup()
    with pytest.raises(ValueError, match="unknown cost axis"):
        make_cost_fn("energy", arch, configs)


def test_cost_axis_is_a_train_config_string():
    TrainConfig(cost_axis="flops").validate()
    with pytest.raises(TypeError):
        TrainConfig(cost_axis=3).validate()


def test_hypervolume_2d():
    ref = (10.0, 10.0)
    assert hypervolume_2d([(2.0, 3.0)], ref) == 8 * 7
    assert hypervolume_2d([(2.0, 6.0), (5.0, 3.0)], ref) == (5 - 2) * 4 + (10 - 5) * 7
    assert hypervolume_2d([(2.0, 3.0), (5.0, 4.0)], ref) == 8 * 7  # second point is dominated
    assert hypervolume_2d([(11.0, 1.0), (1.0, 10.0)], ref) == 0.0
    assert hypervolume_2d([], ref) == 0.0
