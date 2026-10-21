# Experiment status

Written for: the authors. Last updated: 2026-10-13.

E0, E1 and E2 are done for SmolLM2-360M. **The E3 50M-token pilot is done** (job 27783, 2x A40): one run, one seed, untuned lr, no naive-recipe baseline yet. E1/E2 are the baseline that joint nested training has to beat.

## Experiment matrix

Hardware where run: DAS-6 `node205` (NVIDIA A40) unless stated. The Jetson Orin Nano has not been touched.

| ID | Experiment | Config / script | Status | Evidence |
|---|---|---|---|---|
| E0 | Model facts from the hub | `scripts/check_models.py` | **DONE** (18 of 24 matched; Llama 3.2 and Gemma gated; SmolLM3 unrecognized by transformers 4.51.1) | `results/e0_model_facts_2026-10-13.json` |
| E1 | Quality per config, no training, original order vs importance-ordered | `scripts/run_e1_e2.py` | **DONE for SmolLM2-360M** (WikiText-2 and C4 validation, bf16, flash attention) | `results/e1e2_smollm2-360m_flash_attention_2.{json,md}` |
| E2 | Permutation equivalence | same | **DONE for SmolLM2-360M**: max fp32 logit change 5.1e-4 (logits near 30), within tolerance | same |
| E3 | Joint nested training with distillation vs naive recipe | `configs/experiments/smollm2-360m_nested.yaml`, `NPROC=2 run_experiments.sh train` | **DONE for SmolLM2-360M**: pilot (50M, job 27783) and main run (500M, job 27801) | `results/e3_*_eval.json`, tables below |
| E4 | Ablations: distillation, sandwich, ordering | same, `--set` overrides | NOT STARTED | none |
| E5 | Llama-3.2-1B, then 3B, Qwen and SmolLM series | `MODEL=... run_experiments.sh` | NOT STARTED; Llama needs `HF_TOKEN` | none |
| E6 | Monotonicity penalty and configuration weights | `--set train.mono_weight=0.5`; `train.weight_mode=fixed|hv`, `train.max_weight` | weights IMPLEMENTED and tested on CPU; comparison runs queued (27863-27865), no result yet | none |
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

### E3 pilot result (50M tokens, 1,500 steps, 6,236 s on 2x A40)

Start point is the E1/E2 importance-ordered checkpoint. Perplexity, lower is better. Single run, single seed, lr 1e-4 untuned.

| Config | Active params | WikiText-2, E2 start | WikiText-2, pilot | C4 val, E2 start | C4 val, pilot | FineWeb-Edu val, pilot |
|---|---|---|---|---|---|---|
| f025 | 133.8 M | 4,489 | 91.3 | 4,382 | 81.0 | 66.9 |
| f050 | 212.4 M | 406.5 | 41.1 | 289.9 | 41.4 | 33.2 |
| f075 | 298.9 M | 45.6 | 21.0 | 45.4 | 24.9 | 19.3 |
| f100 (full) | 361.8 M | 12.97 | **13.62** | 17.96 | **18.43** | 14.00 |

- The small configurations are now usable: 49x (f025) and 10x (f050) lower WikiText-2 perplexity than before training.
- **The full model got worse**: +5.0% on WikiText-2 and +2.6% on C4. The 30-step probe showed the same direction, and 50M tokens did not
  remove it. In the training log the f100 cross-entropy rises from about 2.4 early to 2.55-2.65 at the end, which looks like drift from a
  learning rate that is too high for the full configuration, not a bug. This is untested.
- Gaps to the full model on WikiText-2 are still 6.7x (f025), 3.0x (f050), 1.5x (f075).
- What it does not show: that the recipe beats the naive one (no baseline run), that more tokens fix the regression, or that it holds on
  another model or seed. Follow-up runs (naive baseline 27797, finished, see below; lr 3e-5 and 1e-5 sweep 27798, 27799; all 50M tokens, same seed and data) were submitted 2026-10-14 and have no results yet. A heavier max-config weight is not implemented. These runs are complete; the 500M main run is next, pending the authors' confirmation of its settings.

### E3 naive-recipe baseline (same 50M tokens, lr 1e-4, one seed)

One random configuration per step and no distillation, so each step costs about a third of the pilot's compute: the comparison is at equal
tokens, not equal FLOPs. Perplexity:

| Config | WikiText-2 pilot / naive | C4 val pilot / naive | FineWeb-Edu pilot / naive |
|---|---|---|---|
| f025 | 91.3 / 156.1 | 81.0 / 110.1 | 66.9 / 92.8 |
| f050 | 41.1 / 50.4 | 41.4 / 44.3 | 33.2 / 35.8 |
| f075 | 21.0 / 24.1 | 24.9 / 25.7 | 19.3 / 20.0 |
| f100 | 13.62 / 15.34 | 18.43 / 18.78 | 14.00 / 14.19 |

The sandwich + distillation recipe is better than the naive one at every configuration and on every set at equal tokens, including the full
model, whose drift is smaller (+5.0% vs +18% on WikiText-2). The naive baseline's full model drifts more, so the drift is not caused by
distillation alone. This is one seed and the naive run saw one third of the forward passes.

### E3 learning-rate sweep (50M tokens, sandwich + distillation, one seed)

Everything except the peak learning rate is the pilot's recipe. WikiText-2 perplexity (C4 and FineWeb-Edu show the same pattern; see
`results/e3_*_eval.json`). All experiments in this project use one seed (seed 0) by the authors' rule.

| Config | lr 1e-4 (pilot) | lr 3e-5 | lr 1e-5 | naive, lr 1e-4 | untrained E2 start |
|---|---|---|---|---|---|
| f025 | 91.3 | 170.2 | 286.1 | 156.1 | 4,489 |
| f050 | 41.1 | 53.4 | 72.0 | 50.4 | 406.5 |
| f075 | 21.0 | 22.8 | 24.8 | 24.1 | 45.6 |
| f100 (full) | 13.62 | 13.44 | 13.32 | 15.34 | 12.97 |

- A lower learning rate reduces the full-model drift (+5.0%, +3.6%, +2.7% over 12.97) but the small configurations recover much less in
  50M tokens (f025: 91, 170, 286). No single peak learning rate gets both at this budget; this is the trade-off that per-configuration
  loss weights would address.
- 50M tokens only: the small-configuration gap may narrow with more tokens, and this has not been tested.

### E3 main run (500M tokens, lr 5e-5, seed 0)

Perplexity (`results/e3_main_smollm2-360m_500M_eval.json`); pilot = 50M tokens at lr 1e-4.

| Config | WikiText-2 main / pilot | C4 val main / pilot | FineWeb-Edu val main / pilot | Untrained E2 start (WikiText-2, C4) |
|---|---|---|---|---|
| f025 | 52.7 / 91.3 | 52.9 / 81.0 | 41.8 / 66.9 | 4,489, 4,382 |
| f050 | 31.0 / 41.1 | 33.6 / 41.4 | 26.2 / 33.2 | 406.5, 289.9 |
| f075 | 18.9 / 21.0 | 23.2 / 24.9 | 17.7 / 19.3 | 45.6, 45.4 |
| f100 (full) | 13.82 / 13.62 | 18.70 / 18.43 | 14.14 / 14.00 | 12.97, 17.96 |

- Small configurations improved with 10x the tokens: f025 on WikiText-2 91 to 53, f050 41 to 31. Gaps to the full model are now 3.8x (f025),
  2.2x (f050) and 1.37x (f075) on WikiText-2.
- **The full model drifted further**: +6.5% on WikiText-2 and +4.1% on C4 over the untrained model, worse than the pilot (+5.0%, +2.6%) and
  than the 50M-token runs at lr 3e-5 and 1e-5. More tokens at a lower peak rate did not remove the drift, so the drift is not only a
  learning-rate effect at this scale.
- Untested explanations, not results: the shared weights trade full-model quality for the small configurations; or a shift from the
  pretraining distribution to FineWeb-Edu (the untrained model's FineWeb-Edu perplexity was never measured, so this cannot yet be separated).
- One seed, one learning rate, one model.

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
| 27783 | **E3 pilot, 50M tokens** | defq | 205 (2x A40) | passed, 1,500 steps, 6,236 s |
| 27784 | CPU tests | fatq | 201 | 245 of 248 passed; 3 test-isolation failures, fixed |
| 27796 | CPU tests after adding `train.sandwich` | fatq | 201 | 249 passed |
| 27797 | **E3 naive-recipe baseline**, 50M tokens: `train.sandwich=false kd_weight=0 teacher=none`, lr 1e-4 (one random config per step, no distillation), run `e3_naive_lr1e-4_50M` | defq | 205 (2x A40) | passed, results below |
| 27798 | E3 recipe sweep, 50M tokens, lr 3e-5, run `e3_sandwich_lr3e-5_50M` | defq | 205 | passed, results below |
| 27799 | E3 recipe sweep, 50M tokens, lr 1e-5, run `e3_sandwich_lr1e-5_50M` | defq | 205 | passed, results below |
| 27850 | CPU tests with config weights, HV outer step, `DEPENDENCY` option | fatq | 201 | submitted 2026-10-19; gates the GPU jobs below |
| 27853 to 27864 | first submission; cancelled before start (dependencies were not honoured), then re-queued as 27875 to 27878 (A10) and 27865 (A40) | | | superseded |
| 27875, 27876 | step 1a (untrained full model, eval only, A10) and step 1b (full model only, CE only, 50M tokens, A10, batch 4 x 4) | defq | 207, 208 | passed |
| 27877, 27878 | equal weights and fixed 3x weight, 50M tokens, lr 5e-5, seed 0 (A10, batch 4 x 4) | defq | 208 | passed |
| 27866 to 27868 | Llama-3.2-1B: E1/E2 (publishes the permuted checkpoint), 30-step probe (36.5 GB peak), 1500-step 50M-token pilot at lr 1e-4 | defq | A40 | passed |
| 27865 | HV-weighted run (first version; see the bug note below), 50M tokens, lr 5e-5 (A40) | defq | 205 | passed |
| 27888 | CPU tests after the HV fixes and re-scoring script | fatq | 201 | passed, 278 tests |
| 27889 | **HV rerun, fixed version** (cosine alignments, full-model floor 1.0, selection batches disjoint from evaluation), 50M tokens, lr 5e-5, seed 0 (A40 x2), run `e3_hvfix_lr5e-5_50M` | defq | 205 | queued behind 27888 |
| 27890 to 27895 | re-scoring on all blocks of each evaluation file, one protocol: untrained model, full-only, equal, fixed 3x, pilot, first HV (excluding its selection blocks) | defq | 207 (A10, one at a time) | chained after 27888 |
| 27851, 27852 | Llama-3.2-1B evaluation data (done) and 520M-token training data | fatq | 202, 203 | 27851 done; 27852 running |
| 27861 | step 1a: untrained SmolLM2-360M full model on FineWeb-Edu val (and WikiText-2 cross-check), bf16, flash | defq | - | queued behind 27850 |
| 27862 | step 1b: full model only, CE only, 50M tokens, lr 5e-5 (`e3_fullonly_ce_lr5e-5_50M`) | defq | - | queued behind 27850 |
| 27863, 27864, 27865 | weighting comparison, 50M tokens, lr 5e-5, seed 0: equal weights, fixed `max_weight=3`, hypervolume-driven (`e3_w_{equal,fixed3,hv}_lr5e-5_50M`) | defq | - | queued behind 27850 |
| 27866, 27867, 27868 | Llama-3.2-1B: E1/E2, 30-step probe (memory and speed), then 50M-token pilot with the SmolLM2 pilot recipe (lr 1e-4) | defq | - | queued; 27867 waits for 27866 and 27852, 27868 for 27867 |
| 27801 | **E3 main run, 500M tokens**: 15,260 steps, lr 5e-5 (midpoint of the lr sweep; authors' choice), 300 warm-up steps, cosine to 10%, save every 500, run `e3_main_smollm2-360m_500M`, 48 h limit, seed 0 | defq | 205 (2x A40) | passed, 63,178 s (17.6 h), results below |
| 27800 | CPU tests after the cost axis, Taylor importance and CPU nodes 201-203 | fatq | 201-203 | 266 passed |

Logs: `/var/scratch/skalra/elastoformer_moo/logs/`. Run directories: `/var/scratch/skalra/elastoformer_moo/runs/`.
Environment: `/var/scratch/skalra/prune_llm` (Python 3.10.18, torch 2.7.0+cu126) plus the `pylibs` overlay (transformers 4.51.1,
flash-attn 2.8.3).

### Step 1 controls, weighting arms and Llama pilot (2026-10-19, one seed)

**Read this first, three evaluation caveats.** (1) The training-time evaluation reads only the first `eval_batches` batches of
each file; the FineWeb-Edu validation file has 4,884 blocks and the reported numbers use 320 of them. (2) The A10 runs used a per-GPU
batch of 4, and evaluation takes its batch size from the data config, so their evaluation covered 163,680 tokens instead of the
303,831 (WikiText-2) or 327,360 (FineWeb-Edu) of the other runs. (3) The Llama pilot was evaluated on 81,840 tokens. Cross-run
differences of a few percent are therefore not reliable until the re-scoring jobs (27890 to 27895) give one protocol for all runs.

| Run (50M tokens, seed 0) | Hardware | WikiText-2, full model | Note |
|---|---|---|---|
| Untrained SmolLM2-360M (E2 start) | A40 | 12.97 (full file) | FineWeb-Edu untrained 13.72 (320 blocks) |
| Full model only, CE only, lr 5e-5 (`fullonly`) | A10 | 13.83 (+6.6%) | 163,680 tokens |
| Equal weights, lr 5e-5 | A10 | 13.26 (+2.2%) | 163,680 tokens |
| Fixed 3x weight on full model, lr 5e-5 | A10 | 13.04 (+0.5%) | 163,680 tokens |
| Pilot, lr 1e-4 | A40 | 13.62 (+5.0%) | full file |
| HV-weighted, first version, lr 5e-5 | A40 | 13.68 (+5.5%) | full file; see bug note |

- **What the controls say.** Training the full model alone, with no small configurations and no distillation, still raises its
  WikiText-2 perplexity (+6.6% on the subset). So sharing weights with the small configurations is not the main cause of the drift;
  the learning rate and the training distribution are the likely causes. On FineWeb-Edu the same model went from 13.72 untrained to
  13.02 after training, which fits a shift toward the training domain. The subsets differ, so this is suggestive only.
- **Fixed 3x weight versus equal weights (same subset):** the 3x weight reduces the full-model drift from +2.2% to +0.5% and costs
  the smallest configuration about 10% (f025 130 to 143). That is the trade-off the weighting was meant to buy.
- **Scale check, Llama-3.2-1B pilot (lr 1e-4, the same recipe as the SmolLM pilot):** the small configurations recover well
  (WikiText-2 perplexity of f025, f050, f075 is 51.7, 29.3, 19.3 against starting points of 3,673, 690, 53). But the full model drifts
  more than SmolLM2 at the same learning rate: 10.99 to 12.98 (+18%) on WikiText-2 and 14.7 to 18.3 (+25%) on C4. The learning rate is
  the likely cause. Memory: peak 36.5 GB per GPU at batch 2 times accumulation 8, which fits an A40 (45 GB) and not an A10 (22 GB).

### Hypervolume weighting: bug found and fixed (2026-10-20)

Two faults in the first version, both corrected in the code:

1. **Selection leaked into the evaluation.** The outer step selected weights on the first two blocks of the FineWeb-Edu file, which
   the reported FineWeb-Edu evaluation also reads. The selection batches now come from the end of the file (`TokenBlocks.tail`), and
   the training code refuses a file too small to hold both. `hv_val_set` names the file.
2. **Raw dot products let the smallest configuration dominate.** In the first run the weights moved to the smallest configuration
   (f025 reached 3.5 by step 1400) and the full model's weight sat at its 0.25 floor. The cause: the training loss and gradient norm
   are largest for the smallest configuration (CE 6.4 against 2.6 for the full model at step 50), so its raw alignments were about
   10x those of the full model. The alignments are now cosines, and the full model's weight never drops below `hv_full_min` = 1.0.

Job 27889 reruns the weighting with these fixes on two A40s (`e3_hvfix_lr5e-5_50M`). The first-version result above is kept for the
record only.

### Re-scoring protocol (jobs 27890 to 27895)

`scripts/rescore.py` (stage `rescore` in `run_experiments.sh`) evaluates every block of each evaluation file with the same settings
for every run: fp32 weights, bf16 autocast, flash attention, batch 8, all four configurations. For a hypervolume-weighted run the
last `hv_val_batches * batch_size` blocks of the selection file are excluded. The untrained model is scored the same way, so each
gap is measured against a common baseline.

### Full-file re-scoring of the 50M-token runs (jobs 27890 to 27895, 2026-10-21, one seed)

All blocks of each file (WikiText-2 297 blocks, C4 256, FineWeb-Edu 4,882; the old HV run excludes its 16 selection blocks),
same settings for every run. Full model (f100) perplexity; change against the untrained model in brackets.

| Run | WikiText-2 | C4 | FineWeb-Edu |
|---|---|---|---|
| Untrained (permuted) | 12.96 | 17.95 | 13.50 |
| Full model only, CE only, lr 5e-5 | 14.16 (+9.3%) | 17.91 (-0.2%) | 13.34 (-1.2%) |
| Equal weights, lr 5e-5 | 13.53 (+4.4%) | 18.33 (+2.1%) | 13.68 (+1.3%) |
| Fixed 3x weight on full model, lr 5e-5 | 13.32 (+2.8%) | 18.10 (+0.8%) | 13.52 (+0.1%) |
| Pilot (lr 1e-4) | 13.62 (+5.1%) | 18.43 (+2.7%) | 13.75 (+1.9%) |
| HV first version (excl. selection blocks) | 13.68 (+5.6%) | 18.53 (+3.2%) | 13.82 (+2.4%) |
| HV fixed version, job 27889 (training-time eval, not re-scored) | 13.46 (+3.9%) | 18.25 (+1.7%) | not available |

Small configurations, WikiText-2 (f025 / f050 / f075): equal 131 / 48 / 22.1; fixed 3x 143 / 51 / 22.8; HV fixed 150 / 46 / 22.5.

- The full-model drift on WikiText-2 is not caused by sharing weights with the small configurations: full-only training drifts most (+9.3%).
- On FineWeb-Edu (the training distribution) sharing does cost the full model: full-only 13.34, equal 13.68, fixed 3x 13.52.
- Fixed 3x weight beats equal weights on the full model on all three sets, at about 10% cost on the smallest configuration.
- HV did not beat fixed 3x. Its weights drifted toward the smallest configuration (f025 at the 0.25 floor, f050 at 2.3 by step 1,400), and the cosine alignments fell to about 0.02 to 0.04 late in training.
- One seed, so differences of about 1 to 2% may be within seed noise. The HV fixed run still needs re-scoring for FineWeb-Edu.

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

**Qwen3 importance-ordering finding, investigated 2026-10-14** (`results/importance_methods_qwen3_smollm2-135m.json`). Protocol: fp32, CPU,
100 calibration blocks of 256 tokens from the text of six of the related papers, 48 held-out blocks from the other four; one calibration
set, no seeds, no training; perplexity. The original smoke check used 6k tokens that overlapped calibration and evaluation.

| Model | Ordering | f025 | f050 | f075 |
|---|---|---|---|---|
| Qwen3-0.6B | original | 3.1e5 | 4,535 | 161 |
| Qwen3-0.6B | activation x weight (current default) | 6.9e6 | 43,908 | 144 |
| Qwen3-0.6B | activation x weight, position 0 excluded | 2.5e6 | 42,066 | 149 |
| Qwen3-0.6B | **first-order Taylor, `|x * dL/dx|`** | **1.1e5** | 4,792 | **111** |
| Qwen3-0.6B | Taylor, position 0 excluded | 1.2e5 | 5,912 | 110 |
| SmolLM2-135M | original | 2.6e7 | 9.3e5 | 1.4e5 |
| SmolLM2-135M | activation x weight | 1,620 | 501 | 202 |
| SmolLM2-135M | Taylor | 1,211 | 450 | 184 |
| SmolLM2-135M | Taylor, position 0 excluded | 1,294 | 329 | 160 |

- The current heuristic is the problem on Qwen3, not the calibration size: with 100 blocks it is still worse than the original order at f025
  and f050, in both its MLP part (mlp-only 5.6e5 / 6.2e4 / 182) and its KV-group part (groups-only 3.5e6 / 3.7e4 / 227).
  The exact equality of the elastic forward and the sliced model is unaffected (that check passed on Qwen3).
- A first-order Taylor score (a backward pass, no parameter gradients) is better than activation x weight on both models: it removes the
  regression at f075 and f025 on Qwen3 and improves on SmolLM2-135M. Excluding position 0 (the attention sink) helps on SmolLM2-135M at
  f050 and f075 and does not matter on Qwen3.
- **Not fixed in the sense of usable:** on Qwen3 the f025 and f050 perplexities stay in the 1e5 and 5e3 range for every ordering, so these
  configurations need training regardless. Taylor on Qwen3 at f050 equals the original order (4,792 vs 4,535), it does not beat it.
- The default is unchanged (`act_weight`), so E1/E2 and the E3 start checkpoint stay valid. Selecting the Taylor score is a flag
  (`permute_model.py --importance-method taylor [--skip-first-token]`). It has not been run on SmolLM2-360M with the C4 calibration set
  or followed by training, so whether it improves E3 is untested.

## Completion evidence

E0, E1, E2 and the attention benchmark are complete for the stated model and hardware. No training run has completed.
