import pytest

from elastoformer_moo import cluster

ALL = [f"node{n}" for n in range(201, 217)]


@pytest.mark.parametrize("node", ["node205", "node206", "node207", "node208"])
def test_gpu_nodes_are_205_to_208(node):
    assert cluster.is_allowed(node, "gpu")
    assert not cluster.is_allowed(node, "cpu")


@pytest.mark.parametrize("node", ["node201", "node202", "node203", "node204"])
def test_cpu_nodes_are_201_to_204(node):
    assert cluster.is_allowed(node, "cpu")
    assert not cluster.is_allowed(node, "gpu")


@pytest.mark.parametrize("node", ["node200", "node209", "node2050", "xnode205", "node205.das6", "login", ""])
def test_everything_else_is_rejected_for_both_kinds(node):
    assert not cluster.is_allowed(node, "gpu") and not cluster.is_allowed(node, "cpu")


def test_exclude_list_leaves_only_the_allowed_gpu_nodes():
    excluded = cluster.exclude_list(ALL, "gpu")
    assert set(ALL) - set(excluded) == {"node205", "node206", "node207", "node208"}


def test_exclude_list_leaves_only_the_allowed_cpu_nodes():
    excluded = cluster.exclude_list(ALL, "cpu")
    assert set(ALL) - set(excluded) == {"node201", "node202", "node203", "node204"}


def test_exclude_list_refuses_when_no_node_matches():
    with pytest.raises(ValueError, match="no valid node|none of the"):
        cluster.exclude_list(["gpu001", "gpu002"], "gpu")


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError, match="kind must be"):
        cluster.is_allowed("node205", "tpu")


def test_patterns_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("ELASTOFORMER_ALLOWED_GPU_NODES", r"gpu0[1-2]")
    assert cluster.is_allowed("gpu01", "gpu") and not cluster.is_allowed("node205", "gpu")
    assert cluster.exclude_list(["gpu01", "gpu02", "gpu03"], "gpu") == ["gpu03"]
