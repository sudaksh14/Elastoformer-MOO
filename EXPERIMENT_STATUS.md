# Experiment status

Written for: the authors. Last updated: 2026-10-13.

E0, E1 and E2 are done for SmolLM2-360M. **E3 is running as a 50M-token pilot** (job 27783, 2x A40); two 30-step probes have run, and
they are probes, not results. E1/E2 are the baseline that joint nested training has to beat.

## Experiment matrix

Hardware where run: DAS-6 `node205` (NVIDIA A40) unless stated. The Jetson Orin Nano has not been touched.

| ID | Experiment | Config / script | Status | Evidence |
|---|---|---|---|---|
| E0 | Model facts from the hub | `scripts/check_models.py` | **DONE** (18 of 24 matched; Llama 3.2 and Gemma gated; SmolLM3 unrecognized by transformers 4.51.1) | `results/e0_model_facts_2026-10-13.json` |
| E1 | Quality per config, no training, original order vs importance-ordered | `scripts/run_e1_e2.py` | **DONE for SmolLM2-360M** (WikiText-2 and C4 validation, bf16, flash attention) | `results/e1e2_smollm2-360m_flash_attention_2.{json,md}` |
| E2 | Permutation equivalence | same | **DONE for SmolLM2-360M**: max fp32 logit change 5.1e-4 (logits near 30), within tolerance | same |
| E3 | Joint nested training with distillation vs naive recipe | `configs/experiments/smollm2-360m_nested.yaml`, `NPROC=2 run_experiments.sh train` | **PILOT RUNNING** (50M tokens, job 27783); the 500M-token main run is not started | probes below |
| E4 | Ablations: distillation, sandwich, ordering | same, `--set` overrides | NOT STARTED | none |
| E5 | Llama-3.2-1B, then 3B, Qwen and SmolLM series | `MODEL=... run_experiments.sh` | NOT STARTED; Llama needs `HF_TOKEN` | none |
| E6 | Monotonicity penalty and configuration weights | `--set train.mono_weight=0.5` | NOT STARTED (penalty implemented; hypervolume weights not) | none |
| E7 | Gradient-conflict measurement | none | NOT STARTED, no code | none |
| E8 | Post-training quantization per configuration | none | NOT STARTED, no code | none |
| E9 | Jetson Orin Nano: TTFT, tok/s, memory, energy | none | NOT STARTED, no code | none |
| E10 | Switching cost, views vs checkpoint reload | `export.py` gives the standalone baseline | NOT STARTED | none |
| E11 | Adaptive controller | none | NOT STARTED, no code | none |
| E12 | Per-configuration low-rank deltas | none | NOT STARTED, no code | none |
| - | Attention backend benchmark (supporting) | `scripts/bench_attention.py` | **DONE** | `results/bench_attention_smollm2-360m_A40.json` |

## E1/E2: SmolLM2-360M, no training

Protocol: bf16, flash attention 2.8.3, sequence length 1024, batch 8. Evaluation data: WikiText-2 raw test (304,978 tokens,
rows joined with `"\n\n"`) and the first 262,144 tokens of C4 `en` validation shard 0. Importance scores from 131,072 tokens
(128 blocks of 1024) of C4 `en` train shard 0. Configurations: keep-fractions 0.25, 0.5, 0.75, 1.0 (KV groups and MLP width
shown). Perplexity, lower is better. Seeds: not applicable (no sampling; first documents of each shard).

| Config | KV groups | MLP width | Active params | WikiText-2, original order | WikiText-2, importance-ordered | C4 val, original | C4 val, importance-ordered |
|---|---|---|---|---|---|---|---|
| f025 | 1 | 768 | 133.8 M | 2.1e7 | 4,489 | 1.7e7 | 4,382 |
| f050 | 3 | 1280 | 212.4 M | 4.8e4 | 406.5 | 7.1e4 | 289.9 |
| f075 | 4 | 2048 | 298.9 M | 179.4 | 45.6 | 223.7 | 45.4 |
| f100 (full) | 5 | 2560 | 361.8 M | 12.96 | 12.97 | 17.96 | 17.96 |

What this shows, and does not:

- Importance ordering helped at every configuration on this model, by orders of magnitude at the smaller ones, and left the full
  model unchanged (12.96 vs 12.97).
- Even the best sub-configuration (83% of parameters) has perplexity 3.5 times the full model on WikiText-2. Without training the
  smaller configurations are unusable. This is the gap E3 has to close.
- The attention backend does not change these numbers: the same run with `sdpa` agrees to 3-4 significant digits
  (`results/e1e2_smollm2-360m_sdpa.json`).
- One model. On Qwen3-0.6B the same importance heuristic made things worse in an earlier smoke check (below); that is open.

## Attention backend benchmark (supporting result)

Setup: A40, SmolLM2-360M, bf16 weights, batch 4, sequence 1024, real WikiText-2 tokens, median of 10 timed iterations after 3 warm-up.
Accuracy is each backend's logit error against the same weights in float32 with `sdpa`.

| Backend | Config | Forward ms | Fwd+bwd ms | Peak GB | Mean abs logit error vs fp32 | Top-1 agreement vs fp32 |
|---|---|---|---|---|---|---|
| sdpa | f025 | 42.3 | 194.1 | 7.18 | 0.176 | 0.886 |
| flash | f025 | 54.6 | 191.7 | 8.49 | 0.177 | 0.887 |
| sdpa | f100 | 75.3 | 253.2 | 10.04 | 0.0548 | 0.978 |
| flash | f100 | 75.2 | 253.3 | 10.91 | 0.0550 | 0.981 |

(f050 and f075 are in the JSON.) Flash and `sdpa` are equally accurate, flash is not faster, and it uses about 1 GB more memory in
this setting. One model, one GPU type, one sequence length; the standalone package may behave differently at other lengths or with
padding-free batches, which were not measured.

## E3 setup and training-data choice

- Corpus: `HuggingFaceTB/smollm-corpus`, subset `fineweb-edu-dedup` (ODC-By, ungated): the same distribution SmolLM2 was pretrained on.
  520,000,000 training tokens from shard 0 and 5,000,000 validation tokens from shard 1 (`scripts/prepare_train_data.py`, 1640 s at about
  340k tokens/s). Evaluation stays on WikiText-2 and C4 validation, which are disjoint from the training corpus. The reasoning, and the
  comparison of corpora and budgets across the related work, is in `research/TRAINING_DATA_COMPARISON.md`.
- Start point: the importance-ordered checkpoint from E2. Recipe: sandwich sampling (largest, smallest, one random configuration), cross-entropy
  plus KL distillation from a frozen copy of the full model, fp32 master weights under bf16 autocast, flash attention, AdamW lr 1e-4 (untuned),
  cosine to 10%, 100 warm-up steps, no frozen parameters.
- Parallelism: 2 processes, one per A40, explicit gradient averaging once per optimizer step. Global batch 32 sequences of 1,024 tokens
  = 32,768 unique tokens per step. Data order: shuffled epochs, disjoint across ranks, exact on resume.
- **Measured throughput: 4.11 s per step, about 8,000 tokens/s on two A40s.** At that rate 50M tokens takes about 1.7 hours and 500M about
  17 hours. This is below my earlier 11k tokens/s estimate.
- Node-local staging of the data to `/local` (NVMe) gave the same 4.11 s/step as reading from NFS, so training is compute-bound.

### 30-step probes (about 1M tokens each; not results)

These check that the pipeline runs end to end. Learning-rate warm-up was 5 steps, so the schedule is not representative.

| Config | WikiText-2, E2 start | WikiText-2, after 30 steps | C4 val, E2 start | C4 val, after 30 steps |
|---|---|---|---|---|
| f025 | 4,489 | 743 | 4,382 | 451 |
| f050 | 406.5 | 141 | 289.9 | 98.2 |
| f075 | 45.6 | 30.3 | 45.4 | 32.9 |
| f100 (full) | 12.97 | 13.4 | 17.96 | 18.4 |

The small configurations improved by 5x to 10x within 30 steps, and the full model got about 3% worse. Whether that cost disappears with more
training, or needs the max-config protection the plan describes, is what the pilot and E4 are for.

## DAS-6 job log

| Job | What | Partition | Node | Result |
|---|---|---|---|---|
| 27762, 27763, 27765, 27767 | smoke tests and first flash-attn build | fatq / defq | 201, 201, 207, 201 | failed (fixed); see `IMPLEMENTATION_STATUS.md` |
| 27764, 27766 | CPU tests, GPU smoke test | fatq, defq | 201, 207 (A10) | passed |
| 27768, 27769 | download SmolLM2-360M, prepare token data | fatq | 202, 203 | passed |
| 27770 | compile flash-attn 2.8.3 (ninja, arch 80) | fatq | 201 | passed, 1278 s |
| 27771 | GPU tests incl. flash | defq | 205 (A40) | 8 passed |
| 27772, 27773 | first benchmarks (random ids; real text but inflated memory) | defq | 205 | superseded by 27776 |
| 27774 | E1/E2, flash attention | defq | 205 (A40) | done |
| 27775 | E1/E2, sdpa (cross-check) | defq | 205 (A40) | done |
| 27776 | attention benchmark, clean | defq | 205 (A40) | done |
| 27777 | CPU tests | fatq | 201 | 231 passed |
| 27778 | download Llama-3.2-1B (HF token) | fatq | 201 | passed, 2.48 GB |
| 27779 | 2-GPU NCCL smoke test | defq | 205 (2x A40) | passed; all-reduce of 1.45 GB: 0.15 s (9.65 GB/s) |
| 27780 | tokenize the training corpus (520M + 5M tokens) | fatq | 201 | passed, 1640 s |
| 27781 | 30-step probe, NFS data | defq | 205 | training ok, 4.11 s/step; final evaluation crashed (flash attention with fp32, fixed) |
| 27782 | 30-step probe, node-local data | defq | 205 | passed, 4.11 s/step, evaluation ok |
| 27783 | **E3 pilot, 50M tokens** | defq | 205 (2x A40) | running |
| 27784 | CPU tests | fatq | 201 | 245 of 248 passed; 3 test-isolation failures, fixed |

Logs: `/var/scratch/skalra/elastoformer_moo/logs/`. Run directories: `/var/scratch/skalra/elastoformer_moo/runs/`.
Environment: `/var/scratch/skalra/prune_llm` (Python 3.10.18, torch 2.7.0+cu126) plus the `pylibs` overlay (transformers 4.51.1,
flash-attn 2.8.3).

## Smoke checks on real checkpoints (2026-10-13, CPU, laptop)

`scripts/check_real_model.py`, about 6k held-out tokens of this repository's own text, same text for calibration and evaluation.
**Not results.**

| Model | Family | KV groups | Elastic forward vs sliced standalone model (float64 / fp32) | Active vs standalone parameters |
|---|---|---|---|---|
| SmolLM2-135M | llama | 3 | exactly equal | equal |
| Qwen2.5-0.5B | qwen2 (q/k/v biases) | 2 | exactly equal | equal |
| Qwen3-0.6B | qwen3 (q/k norm) | 8 | exactly equal | equal |

| Model | Order | f025 | f050 | f075 | f100 |
|---|---|---|---|---|---|
| SmolLM2-135M | original | 3.0e7 | 4.1e5 | 1.1e5 | 76.7 |
| SmolLM2-135M | importance-ordered | 4.8e3 | 430 | 230 | 76.7 |
| Qwen2.5-0.5B | original | 8.0e5 | 1.1e6 | 262 | 72.3 |
| Qwen2.5-0.5B | importance-ordered | 4.8e3 | 727 | 96 | 72.3 |
| Qwen3-0.6B | original | 4.6e5 | 7.6e3 | 268 | 63.0 |
| Qwen3-0.6B | importance-ordered | **8.7e7** | **1.4e5** | **351** | 63.0 |

**Open finding (deferred by the authors):** importance ordering made Qwen3-0.6B much worse in this check. Suspects, not
investigated: per-head q/k norm, massive-activation channels, a tiny calibration set. Settle it before running E1 at scale on Qwen3.
The better-designed E1 on SmolLM2-360M above (131k C4 calibration tokens) shows the heuristic helping.

## Completion evidence

E0, E1, E2 and the attention benchmark are complete for the stated model and hardware. No training run has completed.
