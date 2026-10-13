# Implementation status

Written for: the authors. Last updated: 2026-10-13.

## Summary

- The core library, pipeline stages, tests and cluster tooling are written and tested.
- Tests: **248 passed, 8 skipped** locally (the skips need a GPU). On DAS-6 CPU the suite passed 245 of 248 on the first run; the
  3 failures were a test-isolation bug (the job's own `TIME_LIMIT` leaked into the tests) and are fixed, rerun pending.
  **8 GPU tests passed** on an A40 (job 27771, node205), including flash-attention vs `sdpa` agreement.
- **Flash attention works and is used.** The prebuilt wheel cannot load on DAS-6 (glibc 2.28), so flash-attn 2.8.3 was
  compiled from source into the overlay (job 27770, 21 minutes). E1/E2 ran with `flash_attention_2` as requested.
- **E1 and E2 are done for SmolLM2-360M** (no training): see `EXPERIMENT_STATUS.md`.
- **Multi-GPU training works**: data-parallel with explicit gradient averaging, tested as two real processes on CPU (gloo), and on
  2x A40 with NCCL (job 27779). The **E3 pilot (50M tokens) is running** on node205 (job 27783); no E3 result exists yet.
- Training data is staged to the node-local NVMe (`/local`) at job start: 2 GB in 6 s, no measurable speedup (training is compute-bound).
- Not implemented: quantization, the Jetson runtime and measurements, hypervolume-driven configuration weights,
  gradient-conflict measurement, per-configuration low-rank deltas, downstream benchmarks, training at scale.
  See `IMPLEMENTATION_PLAN.md`.

## Environments

### Local (laptop)

Tests run in the conda environment **`prune_llm`**: Python 3.11.11, torch 2.6.0+cpu, transformers 4.51.1, pytest 9.0.3.
`elastoformer` (transformers 4.28) and `base` (no torch) were rejected.

### DAS-6 (inspected 2026-10-13, output in `results/das6_inspect_2026-10-13.txt`)

| Environment | Python | torch | transformers | flash-attn | Verdict |
|---|---|---|---|---|---|
| `/var/scratch/skalra/prune_llm` (conda prefix) | 3.10.18 | 2.7.0+cu126 | 4.28.0 | 2.8.3 wheel, cannot load (needs glibc 2.32, nodes have 2.28) | **Chosen as the base**, plus an overlay |
| `/var/scratch/skalra/envs/matryoshka-mm` | 3.10.20 | 2.1.2+cu121 | 4.44.2 | 2.3.6, imports fine | Not chosen: transformers 4.44 lacks the attention API this code reuses and Qwen3; torch is older. Its own scripts request `eager` or `sdpa`, not `flash_attention_2` |

Nothing was installed into either shared environment. The overlay `/var/scratch/skalra/elastoformer_moo/pylibs` (built by
`scripts/setup_pylibs.sh` and `scripts/build_flash_attn.sh`) holds transformers 4.51.1, tokenizers 0.21.1, pytest, pyarrow,
ninja and **flash-attn 2.8.3 compiled on the cluster**, and sits first on `PYTHONPATH`.

Cluster facts verified: node names `node201`-`node208`; partitions `fatq` (CPU, node204 was down) and `defq` (GPU: node205
2x A40, node206 1x A40, node207 1x A10, node208 2x A10); `--gres=gpu:1` or `gpu:A40:1`; **no `--mem` ever** (admin rule:
every job occupies the whole node; the nodes also report `RealMemory=1`); Slurm accounting disabled (no `sacct`); home
quota 4 GB soft / 5 GB hard; `/var/scratch` writable; glibc 2.28; GPU jobs need `module load cuda12.1/toolkit/12.1`;
**compute nodes have internet access** (model and dataset downloads worked from fatq nodes); the CPU nodes have no `ninja`.

## Completed

- Repository replaced: the old ViT repository was verified pushed to GitHub (`df028db`), had no ignored or untracked files,
  and was deleted. New git repository at the workspace root, **no commits**. See `MIGRATION.md`.
- Library: `arch`, `config`, `nested`, `importance`, `permute`, `losses`, `sampling`, `data`, `textdata`, `evaluate`,
  `train`, `pipeline`, `e1e2`, `registry`, `export`, `cluster`.
- Scripts: `check_models`, `check_real_model`, `prepare_tokens`, `prepare_eval_data`, `download_model`, `permute_model`,
  `run_e1_e2`, `train_nested`, `eval_ppl`, `bench_attention`, `gpu_smoke`, `build_flash_attn.sh`, `submit.sh`,
  `sync_to_das6.sh`, `setup_pylibs.sh`, `inspect_das6.sh`, `lib.sh`, `node_policy.py`; root `run_*.sh`, `submit_*.slurm`.
- Results recorded in `results/`: E0 model facts, real-checkpoint checks, the DAS-6 inspection, E1/E2 (flash and sdpa),
  and the attention benchmark.
- SmolLM2-360M downloaded, and WikiText-2 / C4 token files prepared on `/var/scratch/skalra/elastoformer_moo`.

## Tests executed (actual outcomes)

| Where | What | Result |
|---|---|---|
| laptop | `run_local_checks.sh` | exit 0; 231 passed, 8 skipped |
| laptop | mutation checks | breaking the MLP slice failed 16 tests; removing the teacher detach failed 2; restored |
| laptop | `check_real_model.py` x3 | elastic forward equals a sliced standalone model exactly (SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B) |
| DAS-6 node201 | job 27777 CPU tests | 231 passed, 8 deselected |
| DAS-6 node205 (A40) | job 27771 GPU tests | smoke ok (CPU/GPU logit diff 2.4e-7); 8 passed incl. 4 flash-vs-sdpa agreement tests |
| DAS-6 node205 (A40) | job 27776 attention benchmark | flash and sdpa equally close to an fp32 reference; flash not faster |
| DAS-6 node205 (A40) | jobs 27774 / 27775 E1/E2 | flash and sdpa quality tables agree to 3-4 significant digits |

Failed jobs along the way (all fixed): 27762 (conda prefix not handled), 27763 (`conda activate` needs the shell
function), 27765 (flash_attn import crash), 27767 (flash-attn build compiled serially because the compute nodes have no
`ninja`; cancelled after 35 minutes at 24 of about 72 kernels, resubmitted as 27770 with ninja installed).

## Bugs found and fixed

- **YAML floats**: PyYAML reads `2e-5` as a string; overrides now parse it and `TrainConfig.validate` rejects wrong types.
- **Teacher hazard**: deep-copying a model with elastic patches made the "frozen teacher" run the student; `build_teacher` detaches first.
- **Verification tolerance**: absolute 1e-4 rejected a correct permutation on a real model; now `atol + rtol * max|logit|`.
  E2 on SmolLM2-360M moved the fp32 logits by 5.1e-4 with logits near 30, which passes.
- **Oracle mismatch**: a 1e-3 difference came from a reference model using `eager` attention instead of `sdpa`; matching kernels give exactly 0.
- **BOM** written by a mutation-check restore; caught by `run_local_checks.sh`.
- **`$USER` unbound** under `set -u`; **conda activation** needs the shell function and nounset off.
- **flash_attn on GPU nodes**: the prebuilt wheel needs a newer glibc than the cluster has, and transformers imports it whenever
  CUDA is visible. First worked around with a metadata shim; now replaced by a real source build in the overlay
  (`scripts/build_flash_attn.sh` removes the shim).
- **Serial flash-attn build** (see above): the build script now installs `ninja` and refuses to continue without it.
- **Evaluation under flash attention** (found by the first 2-GPU probe): the final evaluation ran the float32 model without autocast, and
  flash attention rejects float32, so rank 0 died while rank 1 waited at a barrier until NCCL's 10-minute timeout. Evaluation now uses the
  training autocast, a rank-0 failure is propagated after the barrier, and the NCCL timeout defaults to 90 minutes.
- **Hugging Face token in a tracked file**: a token was placed in `scripts/submit.sh`. It was moved to `~/.hf_token` (outside the repository,
  copied to the cluster over stdin with owner-only permissions), and `tests/test_no_secrets.py` fails if a credential appears in any repo file.
  The token has appeared in this session's transcript and should be rotated at https://huggingface.co/settings/tokens when convenient.
- **Test isolation**: the shell tests inherited the cluster job's `TIME_LIMIT` and asserted on defaults; they now clear ambient submission variables.
- **Benchmark artifacts**: comparing backends on random token ids showed alarming "disagreement" (up to 14 in logits) that was only
  noise amplification; the benchmark now judges each backend against an fp32 reference on real text, and keeps reference logits on
  the CPU so peak-memory numbers are clean.

## Known limits and risks

- **flash-attention gives no speed benefit here.** On an A40 with SmolLM2-360M, batch 4, sequence 1024, bf16: forward is equal or
  slower (54.6 vs 42.3 ms at the smallest config, 75.2 vs 75.3 ms for the full model), training steps are tied, and peak memory is
  about 1 GB higher. PyTorch's `sdpa` already dispatches to a flash kernel in these conditions. It is used for experiments because
  the authors asked for it; it was verified numerically equivalent. This is one model, one GPU, one sequence length.
- **bf16 evaluation is noisy relative to fp32**: for the full model, mean absolute logit error versus fp32 is 0.055 and top-1
  agreement 98%. Quality differences smaller than that scale between configurations should be checked in fp32.
- **transformers coupling**: tested only on transformers 4.51.1.
- **Strided views**: whether PyTorch copies `o_proj`/`down_proj` slices each call is not measured; no latency claim follows.
- **Training is single-device**; the A40 nodes have 1 or 2 GPUs (node205 has 2) and multi-GPU training is not implemented.
- **Resume** is not bit-identical to an uninterrupted run (data stream re-seeded from `(seed, step)`).
- **Open finding**, deferred by the authors: importance ordering made Qwen3-0.6B worse in a no-training check. On SmolLM2-360M it
  helped on every configuration.
- **Gated models** (Llama-3.2-1B and 3B) need `HF_TOKEN`; not yet provided.

## Blockers

1. **`HF_TOKEN`** with the Llama 3.2 license accepted, for the first target model.
2. E3 is running as a 50M-token pilot with untuned hyperparameters (lr 1e-4). The 500M-token main run is not started.

## Revision

- Workspace root: git repository on `main`, **no commits**; all files untracked.
- Old repository: deleted; history on GitHub at `df028db2604c8f8ca2bf4d87af78445e95d30097`.
