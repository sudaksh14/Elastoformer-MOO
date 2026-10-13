# Implementation plan: sensys_2027

Written for: the authors and the implementing engineer.

## 1. Source of the requirements

- `research/SENSYS_2027_PLAN.md` (revised 2026-10-12, SLM scope). It is a research plan, not a software
  specification; its sections 5 to 10 are the source of the requirements below.
- `research/ELASTOFORMER_EXECUTION_PLAN.md` is the journal plan the SLM scope is drawn from.
- Decisions by the authors on 2026-10-13: start with Llama-3.2-1B, then Llama-3.2-3B, Qwen and SmolLM;
  cover the popular small models (TinyLlama, MobileLLaMA and others); a **new, clean repository** replacing
  the old ViT repository, reusing what is reusable, with the old git repository deleted.

## 2. Status legend

`TESTED` = implemented and a test that exercises it ran and passed (CPU unless stated).
`IMPLEMENTED` = code exists, no test has exercised it.
`NOT STARTED` / `BLOCKED` = no code, reason given.

## 3. Requirement map

| ID | Requirement | Files | Status |
|---|---|---|---|
| R1 | Nested configurations: MLP prefix (SwiGLU) and KV-group prefix, head dimension unchanged, shared weights, no copies | `arch.py`, `config.py`, `nested.py` | TESTED (exact in float64 against a sliced standalone model, 4 architecture families, tiny random models; real checkpoints: see EXPERIMENT_STATUS.md) |
| R2 | One-time importance ordering applied consistently to q/k/v/o and gate/up/down, with a logit-equivalence check | `importance.py`, `permute.py` | TESTED |
| R3 | Joint nested training: sandwich sampling, distillation from a frozen full model, no frozen shared weights | `losses.py`, `sampling.py`, `train.py`, `pipeline.py` | TESTED on tiny models (CPU); never run at scale; GPU path IMPLEMENTED, not run |
| R4a | Monotonicity penalty | `losses.py`, `train.py` | TESTED |
| R4b | Hypervolume-driven configuration weights | none | NOT STARTED |
| R4c | Gradient-conflict measurement (E7) | none | NOT STARTED |
| R4d | Per-configuration low-rank deltas (E12) | none | NOT STARTED |
| R5 | Post-training quantization per configuration (8-bit, 4-bit) | none | NOT STARTED |
| R6 | Jetson Orin Nano runtime, latency, memory, energy, switching cost, adaptive controller | none | NOT STARTED: the device and runtime path have not been touched |
| R7 | Experiments E0 to E12 | `scripts/`, `configs/experiments/`, `e1e2.py` | E0, E1, E2 DONE for SmolLM2-360M (no training); E3 pilot DONE (50M tokens, no baseline yet); E4 onward NOT STARTED, see EXPERIMENT_STATUS.md |
| R8a | Perplexity per configuration | `evaluate.py`, `data.py` | TESTED |
| R8b | Downstream benchmarks (ARC, HellaSwag, PIQA, WinoGrande) | none | NOT STARTED: `lm-eval` is not installed in any inspected environment |
| R9 | Model coverage and registry | `configs/models.yaml`, `registry.py`, `arch.py` | TESTED (registry, unsupported-model errors); llama, mistral, qwen2, qwen3 supported; gemma, phi, phi3, smollm3 raise `UnsupportedArchitectureError` |
| R10 | Cluster tooling with node guards (GPU 205-208, CPU 201-204) | `cluster.py`, `scripts/lib.sh`, `scripts/submit.sh`, `scripts/sync_to_das6.sh`, `scripts/setup_pylibs.sh`, `run_*.sh`, `submit_*.slurm` | TESTED locally (bash tests, dry run) and **on DAS-6**: CPU job 27764 on node201 and GPU smoke job 27766 on node207 both ran on allowed nodes and passed. Experiment submission (`submit_experiments.slurm`) has not been exercised |
| R11 | Independent-checkpoint baseline (a configuration as a standalone model) | `export.py` | TESTED |
| R12 | Replace the old ViT repository with a clean one | whole repo, `MIGRATION.md` | DONE: old repo verified pushed (commit `df028db`), then deleted; new repo initialized, no commits yet |
| R14 | Flash attention as an explicit, never-silent backend | `pipeline.py` (`attn_implementation`), `scripts/build_flash_attn.sh`, `scripts/bench_attention.py`, `tests/test_gpu.py` | TESTED on an A40 (flash vs sdpa agree; elastic forward equals a standalone model under flash). Compiled from source because the prebuilt wheel cannot load on DAS-6. Benchmarked: not faster than `sdpa` for SmolLM2-360M at sequence 1024 |
| R15 | Multi-GPU (data-parallel) training | `distributed.py`, `train.py`, `pipeline.py`, `scripts/dist_smoke.py`, `run_experiments.sh` (`NPROC`) | TESTED: two real processes on CPU (gloo) reproduce one-process training on the combined batch; 2x A40 NCCL smoke test passed |
| R16 | Training corpus and exact-epoch data sampling | `scripts/prepare_train_data.py`, `data.py` (`epoch_stream`), `research/TRAINING_DATA_COMPARISON.md` | TESTED (sampling); corpus prepared on DAS-6 |
| R13 | ViT evaluation extension (optional, section 12 of the plan) | none | NOT STARTED; would need the old repo at `df028db`, see MIGRATION.md |

## 4. Order of remaining work

1. E3: the 50M-token pilot is done (full model regressed 5%; fix and baseline pending); then the 500M-token main run (about 17 hours on 2x A40), then E4 ablations.
2. Llama-3.2-1B needs `HF_TOKEN` (gated). Until then, SmolLM2 and Qwen are the runnable models.
3. R4b to R4d, then R5, then R6 (needs the Jetson and a runtime decision).
4. R8b once `lm-eval` is available in the chosen environment.

Deferred by the authors: the Qwen3 importance-ordering finding (see `EXPERIMENT_STATUS.md`).

## 5. Blockers

- **`HF_TOKEN`** with the Llama 3.2 license accepted (free Hugging Face account; see README).
- **No token files** prepared on the cluster yet.
- DAS-6 access works through the ssh-agent method: once per Git Bash session,
  `eval "$(ssh-agent -a ~/.ssh/agent.sock)" && ssh-add ~/.ssh/id_rsa`, then `scripts/sync_to_das6.sh`.
- No CUDA on the laptop, so the four GPU tests only run on DAS-6.
