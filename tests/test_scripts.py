"""Shell-script behaviour: node guard, submit dry run, environment activation failure.

These need a POSIX bash. They are skipped where none is available (for example plain Windows
without Git Bash). On Windows the Git Bash in 'Program Files\\Git' is used.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def find_bash():
    if sys.platform == "win32":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
        return None
    return shutil.which("bash")


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash available")


# Variables a job or shell may export that change what the scripts do. The tests assert on defaults, so they
# must not inherit these: on DAS-6 the test job itself is submitted with TIME_LIMIT set.
AMBIENT = ("SLURMD_NODENAME", "TIME_LIMIT", "PARTITION", "GPU_GRES", "DRY_RUN", "ALL_NODES", "MODEL", "RUN_ID",
           "NPROC", "ATTN", "STAGE_LOCAL", "STAGE_ROOT", "KEEP_STAGE", "RESUME", "CONFIG", "NODE_OVERRIDE",
           "ELASTOFORMER_ENV", "SBATCH_MEM_PER_NODE")


def run(args, env_extra=None, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT}
    env.update(env_extra or {})
    return subprocess.run([BASH, *args], capture_output=True, text=True, cwd=cwd, env=env, timeout=60)


def guard(kind, node, extra=None):
    snippet = f'source scripts/lib.sh; require_node_class {kind}'
    return run(["-c", snippet], {"NODE_OVERRIDE": node, **(extra or {})})


@pytest.mark.parametrize("node", ["node205", "node206", "node207", "node208"])
def test_guard_accepts_gpu_nodes_for_gpu_work_only(node):
    assert guard("gpu", node).returncode == 0
    refused = guard("cpu", node)
    assert refused.returncode == 3 and "refusing" in refused.stderr


@pytest.mark.parametrize("node", ["node201", "node202", "node203"])
def test_guard_accepts_cpu_nodes_for_cpu_work_only(node):
    assert guard("cpu", node).returncode == 0
    refused = guard("gpu", node)
    assert refused.returncode == 3 and "refusing" in refused.stderr


@pytest.mark.parametrize("node", ["node200", "node209", "node2050", "fs2", "login", "node205x"])
def test_guard_rejects_everything_else(node):
    assert guard("gpu", node).returncode == 3
    assert guard("cpu", node).returncode == 3


def test_guard_agrees_with_the_python_policy():
    from elastoformer_moo import cluster

    for n in range(195, 215):
        node = f"node{n}"
        for kind in ("gpu", "cpu"):
            assert (guard(kind, node).returncode == 0) == cluster.is_allowed(node, kind), (node, kind)


def test_guard_rejects_unknown_kind():
    assert guard("tpu", "node205").returncode == 2


def test_gpu_script_refuses_a_cpu_node_before_doing_anything():
    result = run(["run_gpu_tests.sh"], {"NODE_OVERRIDE": "node201"})
    assert result.returncode == 3
    assert "refusing to run gpu work" in result.stderr
    assert "GPU smoke test" not in result.stderr  # nothing after the guard ran


def test_cpu_script_refuses_a_gpu_node():
    assert run(["run_cpu_tests.sh"], {"NODE_OVERRIDE": "node206"}).returncode == 3


@pytest.mark.parametrize("stage", ["permute", "train", "eval", "e1e2", "bench"])
def test_experiment_gpu_stages_refuse_cpu_nodes(stage):
    result = run(["run_experiments.sh", stage], {"NODE_OVERRIDE": "node203"})
    assert result.returncode == 3 and "refusing" in result.stderr


@pytest.mark.parametrize("stage", ["download", "data"])
def test_experiment_cpu_stages_refuse_gpu_nodes(stage):
    result = run(["run_experiments.sh", stage], {"NODE_OVERRIDE": "node205"})
    assert result.returncode == 3 and "refusing to run cpu work" in result.stderr


def test_experiment_script_rejects_unknown_stage_and_missing_stage():
    assert run(["run_experiments.sh", "bogus"]).returncode == 2
    assert run(["run_experiments.sh"]).returncode == 2


def test_missing_environment_fails_clearly(tmp_path):
    env = {"NODE_OVERRIDE": "node201", "ELASTOFORMER_ENV": "does_not_exist_env", "SCRATCH_ROOT": str(tmp_path / "scratch"),
           "HOME": str(tmp_path)}
    result = run(["run_cpu_tests.sh"], env)
    assert result.returncode == 4
    assert "not found" in result.stderr and "ELASTOFORMER_ENV" in result.stderr


NODES = " ".join(f"node{n}" for n in range(201, 217))


def submit(kind, extra=None):
    env = {"ALL_NODES": NODES, "DRY_RUN": "1", "SCRATCH_ROOT": "/tmp/em_scratch_test", **(extra or {})}
    return run(["scripts/submit.sh", kind, "submit_gpu_tests.slurm"], env)


def test_gpu_submission_excludes_everything_but_205_to_208():
    result = submit("gpu")
    assert result.returncode == 0, result.stderr
    line = next(l for l in result.stdout.splitlines() if "--exclude" in l)
    excluded = line.split("--exclude=")[1].split()[0].replace("\\", "").split(",")
    assert set(excluded) == {f"node{n}" for n in range(201, 217)} - {"node205", "node206", "node207", "node208"}
    assert "--gres=gpu:1" in line


def test_cpu_submission_excludes_everything_but_201_to_203():
    result = submit("cpu")
    line = next(l for l in result.stdout.splitlines() if "--exclude" in l)
    excluded = line.split("--exclude=")[1].split()[0].replace("\\", "").split(",")
    assert set(excluded) == {f"node{n}" for n in range(204, 217)}
    assert "--gres" not in line


def test_gpu_submission_queues_on_the_gpu_partition_and_cpu_on_the_cpu_partition():
    gpu_line = next(l for l in submit("gpu").stdout.splitlines() if "--exclude" in l)
    cpu_line = next(l for l in submit("cpu").stdout.splitlines() if "--exclude" in l)
    assert "--partition=defq" in gpu_line and "--partition=fatq" in cpu_line
    assert "--mem" not in gpu_line and "--mem" not in cpu_line  # DAS-6 nodes report RealMemory=1
    assert "--time=00:20:00" in gpu_line and "--time=00:30:00" in cpu_line


def test_no_memory_request_anywhere_on_das6():
    """Admin rule: every DAS-6 job occupies the whole node, so --mem must never appear."""
    for path in [*ROOT.glob("submit_*.slurm"), *ROOT.glob("scripts/*.sh"), *ROOT.glob("run_*.sh")]:
        text = path.read_text()
        for line in text.splitlines():
            code = line.split("#", 1)[0] if not line.lstrip().startswith("#SBATCH") else line
            assert "--mem" not in code, f"{path.name}: {line.strip()}"
    # and an inherited environment variable must not sneak one into sbatch
    result = run(["scripts/submit.sh", "cpu", "submit_cpu_tests.slurm"],
                 {"ALL_NODES": NODES, "DRY_RUN": "1", "SCRATCH_ROOT": "/tmp/em_scratch_test", "SBATCH_MEM_PER_NODE": "8G"})
    assert result.returncode == 0 and "--mem" not in result.stdout


def test_build_jobs_require_an_explicit_time_limit():
    bare = run(["scripts/submit.sh", "cpu", "submit_build_flash_attn.slurm"], {"ALL_NODES": NODES, "DRY_RUN": "1"})
    assert bare.returncode == 2 and "TIME_LIMIT" in bare.stderr
    ok = run(["scripts/submit.sh", "cpu", "submit_build_flash_attn.slurm"],
             {"ALL_NODES": NODES, "DRY_RUN": "1", "TIME_LIMIT": "04:00:00", "SCRATCH_ROOT": "/tmp/em_scratch_test"})
    assert ok.returncode == 0 and "--partition=fatq" in ok.stdout and "--time=04:00:00" in ok.stdout


def test_partition_and_gres_can_be_overridden():
    line = next(l for l in submit("gpu", {"GPU_GRES": "gpu:A40:1", "PARTITION": "defq", "TIME_LIMIT": "00:45:00"}).stdout.splitlines()
                if "--exclude" in l)
    assert "--gres=gpu:A40:1" in line and "--time=00:45:00" in line


def test_experiment_jobs_require_an_explicit_time_limit():
    result = run(["scripts/submit.sh", "gpu", "submit_experiments.slurm", "train"],
                 {"ALL_NODES": NODES, "DRY_RUN": "1", "SCRATCH_ROOT": "/tmp/em_scratch_test"})
    assert result.returncode == 2 and "TIME_LIMIT" in result.stderr
    ok = run(["scripts/submit.sh", "gpu", "submit_experiments.slurm", "train"],
             {"ALL_NODES": NODES, "DRY_RUN": "1", "SCRATCH_ROOT": "/tmp/em_scratch_test", "TIME_LIMIT": "12:00:00"})
    assert ok.returncode == 0, ok.stderr


def test_submission_refuses_when_no_node_is_allowed():
    result = run(["scripts/submit.sh", "gpu", "submit_gpu_tests.slurm"],
                 {"ALL_NODES": "gpu001 gpu002", "DRY_RUN": "1"})
    assert result.returncode == 3 and "not submitting" in result.stderr


def test_submission_rejects_bad_kind_and_missing_script():
    assert run(["scripts/submit.sh", "tpu", "submit_gpu_tests.slurm"], {"ALL_NODES": NODES}).returncode == 2
    assert run(["scripts/submit.sh", "gpu", "nope.slurm"], {"ALL_NODES": NODES}).returncode == 2


def test_all_shell_scripts_parse():
    scripts = [*ROOT.glob("*.sh"), *ROOT.glob("scripts/*.sh"), *ROOT.glob("submit_*.slurm")]
    assert len(scripts) >= 9
    for script in scripts:
        result = run(["-n", script.relative_to(ROOT).as_posix()])
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_stage_data_copies_to_local_storage_and_cleans_up_on_exit(tmp_path):
    src, dest = tmp_path / "shared", tmp_path / "local" / "run"
    src.mkdir()
    (src / "train.bin").write_bytes(b"x" * 5000)
    (src / "val.bin").write_bytes(b"y" * 100)
    script = (f'source scripts/lib.sh; stage_data "{src.as_posix()}" "{dest.as_posix()}" train.bin val.bin '
              f'&& echo "STAGED=$STAGED_DIR" && ls "$STAGED_DIR"')
    result = run(["-c", script])
    assert result.returncode == 0, result.stderr
    assert f"STAGED={dest.as_posix()}" in result.stdout and "train.bin" in result.stdout and "val.bin" in result.stdout
    assert "staged" in result.stderr
    assert not dest.exists()  # removed when the shell exited


def test_stage_data_keep_and_reuse(tmp_path):
    src, dest = tmp_path / "shared", tmp_path / "local"
    src.mkdir()
    (src / "a.bin").write_bytes(b"1" * 2000)
    base = f'source scripts/lib.sh; stage_data "{src.as_posix()}" "{dest.as_posix()}" a.bin'
    assert run(["-c", base], {"KEEP_STAGE": "1"}).returncode == 0
    assert (dest / "a.bin").read_bytes() == b"1" * 2000
    (src / "a.bin").write_bytes(b"2" * 2500)  # the source changed size: the stale copy must be replaced
    assert run(["-c", base], {"KEEP_STAGE": "1"}).returncode == 0
    assert (dest / "a.bin").read_bytes() == b"2" * 2500


def test_stage_data_failure_falls_back_instead_of_aborting(tmp_path):
    src = tmp_path / "shared"
    src.mkdir()
    (src / "a.bin").write_bytes(b"1")
    missing = run(["-c", f'source scripts/lib.sh; stage_data "{src.as_posix()}" "{(tmp_path / "d").as_posix()}" nope.bin'])
    assert missing.returncode == 1 and "missing" in missing.stderr
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")  # a path whose parent is a file cannot be created
    unwritable = run(["-c", f'source scripts/lib.sh; stage_data "{src.as_posix()}" "{(blocker / "sub").as_posix()}" a.bin'])
    assert unwritable.returncode == 1 and "cannot create" in unwritable.stderr


def test_dependency_option_is_passed_to_sbatch_and_absent_by_default():
    assert "--dependency" not in submit("gpu").stdout
    line = next(l for l in submit("gpu", {"DEPENDENCY": "afterok:1:2"}).stdout.splitlines() if "--dependency" in l)
    assert "--dependency=afterok:1:2" in line
