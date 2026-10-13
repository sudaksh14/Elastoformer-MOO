# Test plan

Written for: the authors and the implementing engineer.

Actual outcomes are in `IMPLEMENTATION_STATUS.md`. A test is called passed only if it ran and returned exit 0.

## 1. Local lightweight tests (any machine, no GPU, no dataset, no cluster)

```bash
./run_local_checks.sh                 # PYTHON=/path/to/python to choose the interpreter
# or directly:
python -m pytest -p no:cacheprovider
```

`run_local_checks.sh` does three things: parses every `.py` with `ast` (no files written), runs `bash -n` on
every shell and Slurm script, and runs pytest. Expected: `... passed, 8 skipped` (4 GPU tests and 4 flash-attention tests need a GPU). Last run: 248 passed, 8 skipped.
The distributed tests spawn two processes and take about a minute.
Fails (non-zero) on any syntax error, shell syntax error, or failing test.

| Group | File | What it checks |
|---|---|---|
| Configuration | `test_config.py` | architecture facts, unsupported-model errors, grid ordering and rounding, validation messages |
| Elastic forward | `test_nested.py` | full config bit-identical; every config equals a physically sliced standalone model (4 families); parameter counts exact; `state_dict` keys unchanged; KV-cache generation equals the standalone model; gradients reach only the active prefix |
| Exactness | `test_exact_float64.py` | in float64 (RMSNorm kept in float64) the elastic forward and permutation differ from the oracle by at most 1e-12 / 1e-10; verification catches a half-applied permutation |
| Permutation | `test_permute.py` | function preserved; importance order respected; failed check restores the weights; hooks removed |
| Losses and sampling | `test_losses_sampling.py` | KL formula, next-token cross-entropy, monotonicity penalty, sandwich sampler determinism |
| Training | `test_train.py` | teacher isolation (the deepcopy hazard), micro-step, full loop, resume, config type validation |
| Data, registry, evaluation | `test_data_registry_eval.py` | token blocks, registry flags backed by recorded hub facts, perplexity |
| Pipeline | `test_pipeline.py` | permute, train and evaluate a tiny saved checkpoint end to end; override parsing |
| Export | `test_export.py` | standalone model parameter count, save and load, copy semantics, agreement with the independent oracle |
| Cluster policy | `test_cluster.py` | which node names are allowed for GPU and CPU work |
| Distributed | `test_distributed.py` | gradient averaging and collectives; divergence detection; two processes reproduce one process on the combined batch (weights and pre-clip gradient norm); resume across ranks; `run_training` as two torchrun-style processes |
| Secrets | `test_no_secrets.py` | no token or key in any repository file |
| Shell | `test_scripts.py` | node guard for GPU and CPU scripts, submission dry run, missing-environment failure |

`test_scripts.py` needs a POSIX bash (Git Bash on Windows) and is skipped without one.

## 2. CPU-only tests on DAS-6 (nodes 201-204)

```bash
scripts/submit.sh cpu submit_cpu_tests.slurm
```

Runs `run_cpu_tests.sh`: node guard, environment activation, `pytest -m "not gpu"` with `CUDA_VISIBLE_DEVICES=""`.
Partition `fatq`; time limit 30 minutes (2x a pessimistic 15-minute estimate; override with `TIME_LIMIT`). Exit codes:
3 wrong node, 4 environment missing or cannot import torch/transformers, 5 scratch directory problem.
**Run:** job 27777 on node201, `231 passed, 8 deselected` in 12 s.

One-time setup on the cluster: `scripts/sync_to_das6.sh` (copy the repo), then on the login node
`bash scripts/setup_pylibs.sh` (transformers overlay), then compile flash-attn on a CPU node:
`TIME_LIMIT=02:00:00 scripts/submit.sh cpu submit_build_flash_attn.slurm` (about 21 minutes; installs `ninja` itself).

## 3. GPU tests on DAS-6 (nodes 205-208)

```bash
scripts/submit.sh gpu submit_gpu_tests.slurm
```

Runs `run_gpu_tests.sh`: node guard (no override), `nvidia-smi -L`, `scripts/gpu_smoke.py` (CUDA visible, elastic
logits match CPU for every config, one nested training step with bf16 autocast), then `pytest -m gpu`
(`test_gpu.py`: GPU-vs-CPU logits for 4 families). Partition `defq`, `--gres=gpu:1`; time limit 20 minutes (2x a
pessimistic 10-minute estimate). The script loads `cuda12.1/toolkit/12.1` and checks `torch.cuda.is_available()`
before anything else, because a GPU job that silently runs on CPU is the failure to rule out first. Exit code 6 means
no GPU is visible to nvidia-smi or torch.
**Run:** job 27771 on node205 (A40): smoke ok, CPU/GPU logit difference 2.4e-7, `8 passed, 231 deselected` (includes the flash-attention
agreement tests, which skip themselves if `flash_attn` cannot be imported).
The smoke test uses a tiny random model; it does not measure memory fit or throughput.

## 4. Real-checkpoint checks (need network, CPU is fine for models up to about 1B)

```bash
python scripts/check_models.py --out results/e0_model_facts_<date>.json
python scripts/check_real_model.py --model smollm2-135m --text README.md research/*.md --out results/real_model_check_smollm2-135m.json
```

These compare against real weights and need downloads, so they are not part of the unit-test suite.

## 5. Experiment runs

See `EXPERIMENT_STATUS.md`. Submit one stage at a time and read its log before the next:

```bash
MODEL=smollm2-360m scripts/submit.sh gpu submit_experiments.slurm permute
MODEL=smollm2-360m TIME_LIMIT=04:00:00 scripts/submit.sh gpu submit_experiments.slurm train
```

## 6. Acceptance rules

- Passed means the command ran and exited 0. A planned test that has not run is listed as not run.
- Smoke results are never reported as experiment results.
- Performance and energy numbers come only from measured runs on the stated hardware.
