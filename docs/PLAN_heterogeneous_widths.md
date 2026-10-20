# Plan: hardware-aware heterogeneous per-layer widths on top of Elastoformer-MOO

Plan only. Nothing has been implemented, run or installed.
Labels: **[F]** = verified by reading source/docs this session, **[R]** = reported by a doc/README but not re-run, **[X]** = read second-hand from external repos via a summarising fetch tool (not raw clones), **[P]** = proposal/hypothesis.

## 1. Context

The repo already trains one set of shared weights that serves several nested operating points of a pretrained decoder-only LM (SmolLM2-360M so far). Every operating point applies the *same* width to every layer. The goal is to ask whether choosing *different widths per layer*, selected for a target device and multiple objectives (optionally by hypervolume), gives better quality/latency/memory trade-offs than the uniform nested levels, and how to make that trainable and deployable. Novelty is **not** claimed (see section 9 and Q-lit).

## 2. Repository architecture summary

**Entry points / layout [F]** (`src/elastoformer_moo/`):
- `arch.py` `ArchInfo`, `SUPPORTED_MODEL_TYPES = llama, mistral, qwen2, qwen3` (`arch.py:10`); phi3/gemma/smollm3 rejected; `pretraining_tp != 1` rejected.
- `config.py` `ElasticConfig(name, kv_groups, mlp_width)` frozen dataclass (`config.py:12-52`), `make_grid` (`:60`), `size()` orders configs by elastic parameter count.
- `nested.py` `ElasticController.attach()` monkey-patches instance-level `forward` of every `self_attn` and `mlp` (`:136-170`); `_elastic_attention` (`:51-104`), `_elastic_mlp` (`:107-112`); `ctrl.use(cfg)` context manager; slices are views, `state_dict` keys unchanged.
- `importance.py` (act_weight default, taylor optional), `permute.py` (joint permutation of gate/up rows with down columns, q/k/v rows with o columns by whole KV group; fp32 equivalence check).
- `train.py` `TrainConfig`, `nested_micro_step`, `train`, `_hv_outer_step`; `sampling.py` `SandwichSampler` (largest + smallest + `n_random` from the middle, or naive); `losses.py` CE / KD (T²) / monotonicity penalty; `distributed.py` once-per-step bucketed all-reduce.
- `cost.py` `make_cost_fn` (axes `params`, `flops`, `kv_cache`, `table:PATH` with measured values), `hypervolume_2d`, `hv_widths`; `weights.py` `ConfigWeights` (uncommitted).
- `pipeline.py` (`permute_pretrained`, `run_training`, `apply_overrides` dotted `a.b.c=value`), `evaluate.py` (perplexity, per-config rows), `export.py` `materialize(model, cfg)` builds a physically smaller HF model (`:17`).
- Config system: YAML dict (`configs/experiments/*.yaml`, `configs/models.yaml`) into `TrainConfig(**cfg["train"])`. Checkpoint: HF `final/` via `save_pretrained`, `trainer_state.pt`, `configs.json`, `costs.json`, `eval.json`, `permutation.pt`.
- Uncommitted work in the tree (git status): `cost.py`, `pipeline.py`, `train.py`, new `weights.py`, `tests/test_weights.py`, scripts. This is "R4b" *loss-weighting* by hypervolume sensitivity, **not** architecture search. Any new work should rebase on it or wait for it to land.

**What "width" means here [F]:** the residual hidden size `H` (960 for SmolLM2-360M) is never changed. The two knobs are `kv_groups` (g; query heads `g*r`, `r = num_heads/num_kv_heads`, KV heads `g`; `head_dim` constant) and `mlp_width` (SwiGLU intermediate channels). Both are **global**: one pair per config, identical in all 32 layers. Per-layer widths, residual-width elasticity and depth elasticity do not exist.

**Tensor shapes [F]** (`nested.py`): `q = x @ Wq[:g*r*d].T -> [B,T,g*r,d]`; `k,v = ... Wk/Wv[:g*d] -> [B,T,g,d]`; Qwen3 `q_norm/k_norm` over `head_dim` applied; RoPE via the model's own `apply_rotary_pos_emb` with the model's `position_embeddings`; `cache.update(k,v,layer_idx)` stores `g` heads; HF attention interface does GQA expansion; output `reshape(B,T,g*r*d) @ Wo[:, :g*r*d].T -> [B,T,H]`. MLP: `gate,up = x @ W[:w].T -> [B,T,w]`, `down = (act(gate)*up) @ Wd[:, :w].T -> [B,T,H]`. Norms, embeddings, lm_head (tied for SmolLM2) untouched.

**Not present [F]:** ViT code (deleted; history at `df028db`, `paper/.../extension_vit.tex` is text only); `torch.compile`; CUDA graphs; quantization; Jetson runtime; any latency/energy measurement beyond `scripts/bench_attention.py` (A40 sdpa vs flash). Runtime switching = Python-level `ctrl.use()`. `cost.table:` axis can ingest measured numbers but none exist. Sampling and the monotonicity penalty order configs by parameter count, not by the selected cost axis (`cost.py` docstring).

**Existing evidence [R/F]:** E1-E3 done for SmolLM2-360M, single seed. 500M-token run (17.6 h, 2xA40): WikiText-2 ppl f025/f050/f075/f100 = 52.68 / 30.99 / 18.93 / 13.82 [F, read from results JSON]; full model drifted +6.5% vs original. Sandwich beats naive at equal tokens (not equal FLOPs). Qwen3-0.6B: act_weight ordering is worse than original at f025/f050; Taylor only partly fixes this; only SmolLM2 is a validated base. Stale docs: README/IMPLEMENTATION_STATUS still say "no commits / hv not implemented".

## 3. Comparison matrix (repo vs SHARD vs FlexSLM vs ViT-Slim)

SHARD/FlexSLM/ViT-Slim columns are **[X]**: second-hand, from fetched README/source summaries; FlexSLM `flex_modules/*`, SHARD Jetson harness and ViT-Slim `get_sparsity_loss` were not read. Re-verify by cloning before relying on any cell.

| Aspect | This repo | SHARD [X] | FlexSLM [X] | ViT-Slim (2201.00814) [X] |
|---|---|---|---|---|
| Model family | decoder LM (Llama/Qwen families) | DeiT-III ViT-B/16 | LLaMA-style LM | DeiT ViT |
| Shared weights / nesting | yes, prefix views of one HF model | yes, prefix slices per layer | yes, nested levels | no: gates then one pruned static model |
| Width dimension | KV groups + MLP channels; **residual fixed** | residual D + head size (heads fixed) + MLP, plus embeddings/norms/head | per-level hidden, heads, KV heads, intermediate (residual sliced) | per-unit gates on heads/channels, MLP units, tokens; residual not searched |
| Width choices | 4 global levels (f025..f100) | 5 global levels | few global levels | per-layer non-uniform via global threshold |
| Per-layer heterogeneity | no | no | no | yes (but no nesting) |
| Training / sampling | sandwich: largest+smallest+random middle; grads accumulated, one step | widest + smallest + 2 random middle | smallest+largest+random middle | search: learn gates (L1), then retrain pruned |
| Distillation | frozen copy / self_full KD | progressive (each level from next wider) or from widest | frozen external teacher KD | optional DeiT-style |
| Runtime switching | Python slice, `ctrl.use` | delta manager (up/down deltas), 19.9 ms vs 439 ms reload (claimed) | same delta mechanism (timing script has unit bug) | none |
| Hardware metrics | none (cost axis hook only) | Jetson Orin latency/power/energy (claimed) | switch time only | none |
| Search | none | none (manual levels) | none | continuous gradient search, single-shot |
| Multi-objective / HV | HV for loss weights only | no | no | single budget |

**Reuse:** the nested training/sandwich/KD structure (already here); SHARD's delta-switching and Jetson harness *ideas* (E10/E9); ViT-Slim's idea of a global budget threshold across layers as a cheap baseline. **Genuinely different if built:** per-layer width vectors *inside* a nested supernet + hardware-measured objectives + Pareto/HV selection. **Not established:** that this combination is novel; ViT-Slim already does non-uniform allocation, and the supernet-plus-search family (once-for-all / hardware-aware transformer NAS / MatFormer-style nested FFN / Sheared-LLaMA-style structured pruning) is large. Those names are from memory and unverified; a proper literature review is a gating task (M0).

## 4. Shape-validity invariants (for per-layer widths)

Hold fixed globally: `hidden_size H`, `head_dim d`, `num_layers`, vocab, embeddings, `lm_head` (tied: untouched, so tying stays valid), all norm weights (size H), RoPE tables (depend on `d` and positions only).
Per layer *l*, free variables:
1. `mlp_width_l`: any value in `[1, I]`, deployed as a multiple of 128 or 256 (tensor-core / vectorised kernel alignment; the repo already uses 256). Residual add is valid for any width because `down_proj` output is always `H`.
2. `kv_groups_l` (stage 2): integer in `[1, num_kv]`. Must cut whole KV groups so GQA ratio `r` stays constant: q heads = `g_l*r`, kv heads = `g_l`. `head_dim` unchanged keeps RoPE and Qwen3 q/k-norm valid. `o_proj` columns sliced to `g_l*r*d`.
3. KV cache: shape per layer `[B, g_l, T, d]`. HF `DynamicCache` allocates per layer on first update, so heterogeneous `g_l` should work; **static/pre-allocated caches and any code assuming one `num_key_value_heads` may not**. Hypothesis to test early, not a fact. Cache memory becomes `sum_l g_l`.
4. Order must stay prefix-nested: the per-layer importance permutation already orders each layer's channels/groups, so any per-layer prefix is consistent with the training ordering.
5. Not interchangeable: changing `H` between blocks needs interface projections (residual mismatch), changes embeddings/norms/head, and is a separate high-risk track (section 11, Option C). Attention width and MLP width are independent knobs and should stay separately encoded.

## 5. Candidate encoding and search

**Encoding [P]:** per-layer integer level indices. `mlp_level_l in {0..K-1}`, width = `round_to_multiple(frac_k * I, 256)` with K=8 fractions (0.25, 0.375, ..., 1.0) giving about 8^32 profiles for SmolLM2 (too large to enumerate, fine for search). Stage 2 adds `attn_level_l in {1..num_kv}` (5 values for SmolLM2). Profile = tuple of ints, deterministic string id (hash of the tuple), no floats. Validity enforced by construction (levels map to aligned widths); a `validate()` rejects anything else. Optional constraint: monotone-in-depth or smoothness regularisation as a search-space prior (ablate).
Implementation shape [P]: new frozen dataclass `LayerwiseConfig(name, mlp_widths: tuple, kv_groups: tuple)`; uniform configs become special cases (all entries equal) so all existing configs, tests and checkpoints stay valid. `ElasticController` patches close over the layer index (they already close over per-module state).

**Search algorithm (recommended) [P]:** two-stage.
1. *Cheap proxy:* validation perplexity on a small fixed token set (e.g. ~0.5-1M tokens, fixed batches; the repo already caches validation batches in `pipeline.py`) evaluated through the shared supernet, no training. Cost axes: analytic params/FLOPs/KV bytes (`cost.py`) plus a per-layer **latency lookup table** measured on the target device (latency is not additive in general; verify additivity, otherwise measure the final candidates end-to-end).
2. *Optimiser:* multi-objective evolutionary search (NSGA-II-style) over the integer vector, seeded with uniform levels, sensitivity-greedy and random profiles. Final front re-measured end-to-end on device. Hypervolume enters either as the front-quality metric (always) or as an acquisition/selection criterion (HV-contribution selection, SMS-EMOA-style).
Alternatives and trade-offs:
- *Greedy sensitivity allocation:* per-layer loss delta when shrinking one layer, then marginal-cost-per-damage greedy. Cheap, strong baseline, ignores interactions.
- *Continuous relaxation (ViT-Slim-like gates):* gradient-based, but must be reconciled with nested prefix weights; more invasive.
- *Constrained single objective* (below): simplest for a fixed device budget.
- *Bayesian optimisation / predictors:* worth it only if each evaluation is expensive; proxy eval is cheap enough that evolutionary search probably suffices.

## 6. Evaluation plan and hypervolume

Objectives: quality (val ppl on held-out WikiText-2/C4/FineWeb-Edu, later downstream tasks), latency (prefill and per-token decode, batch 1 and a batch>1 point), peak memory (weights + KV + activations), energy where the device exposes it, params, FLOPs, KV bytes. **Lower FLOPs does not imply lower latency** (the repo itself flags that strided `o_proj`/`down_proj` slices may be copied and has not measured it); report FLOPs only as a secondary axis and make latency the primary hardware axis. Also report compile/capture overhead and profile switching cost separately (not in the front).
Measurement protocol: warm-up, fixed clocks/power mode, sync before timing, many repeats, report median and IQR, pin batch/seq, record driver/torch versions. Noise floor must be measured first (repeat the same profile) and candidates closer than the noise floor must not be ranked.
HV: use 2-3 objectives only (quality, latency, optionally memory). Normalise each objective to [0,1] using fixed anchors from the full model and the smallest feasible profile measured once (not per-run extremes); transform ppl as log-ppl. Reference point fixed *before* search, slightly worse than the worst acceptable point (for example 1.1x of the worst anchor on each axis), identical across all compared methods, and reported alongside results. Report HV with its reference point, the front plot, and a budget-matched comparison (equal number of proxy evaluations / equal wall time). Existing `hypervolume_2d` covers 2-D; a 3-D HV (or a library) is a new dependency decision.
Constrained alternative: minimise val loss s.t. latency <= L_dev, memory <= M_dev (and optionally energy), evolutionary search with feasibility rules; report best quality per budget. This answers the fixed-device-budget question directly and needs no reference point.

## 7. Staged training / search strategy

Stage 0 no-train feasibility: per-layer sensitivity curves on the current E3 checkpoint (shrink one layer at a time), plus random heterogeneous profiles, to see whether heterogeneity matters at all before training anything.
Stage 1 supernet training with profile sampling: extend sandwich to {all-max, all-min, k random heterogeneous profiles}; random profiles drawn with correlated structure (random per-layer levels, plus smooth-in-depth and sensitivity-biased ones) so narrow and wide coverage is maintained and the cost-per-step stays near the current budget. Keep KD (frozen/full-self teacher). Warm-start from the E3 500M checkpoint rather than from scratch. Track per-profile loss, with a monotonicity check along dominance order.
Stage 2 search on the shared supernet using the proxy; Stage 3 target-device measurement of the front; Stage 4 extraction (`materialize` generalised to per-layer, contiguous weights) and optional short fine-tune of selected profiles (compare "extracted, no tune" vs "extracted + tune" vs "independent from scratch/pruned baseline"). Distillation of a profile from the full model, not from a neighbour, is the default; progressive KD (SHARD-style) is an ablation.

## 8. Graph compilation and CUDA Graphs

Per-config shapes are different, so [P]: treat each *selected* profile as a static-shape model. Deployment = extract to a standalone, contiguous model (avoids strided-view copies) and `torch.compile` / capture a CUDA graph per profile, per (batch, seq-bucket). Runtime "arbitrary width change" via slicing views is incompatible with graph replay (pointer/shape capture) and with `torch.compile` without recompiles, so do not promise it. A small supported set of profiles (3-6) with pre-captured graphs, sharing one weight pool or deltas, is the recommended runtime design; measure graph memory per profile, compile time, and switch cost (graph select vs weight load). Verify on target hardware; none of this exists in the repo today.

## 9. Baselines

(a) full model; (b) uniform nested levels (current f025-f100); (c) independently pruned/trained uniform models at matched size (materialize + fine-tune); (d) random heterogeneous profiles at matched cost; (e) greedy sensitivity allocation; (f) ViT-Slim-style global-threshold allocation from the same sensitivity scores; (g) evolutionary search with scalar objective (constrained); (h) evolutionary/HV-guided Pareto search. Same supernet, same evaluation budget, same measurement protocol, and at least 3 seeds for any claim (repo has so far used one seed; this needs a decision, Q5).

## 10. Phased implementation plan (plan only; files are proposals)

Dependency order M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6.

**M0 Groundwork (no code change to model):** literature review (novelty and prior art); clone and read SHARD/FlexSLM raw source; land or branch around the uncommitted R4b changes; reconcile stale docs. *Success:* related-work table with citations; decision recorded on ViT scope.
**M1 Per-layer config representation:** `config.py` (`LayerwiseConfig`, validate, size, id, uniform adapter), `arch.py` untouched. *Tests:* `tests/test_config.py` (validity, alignment, round-trip, determinism, uniform equivalence). *Success:* every existing test passes unchanged; a uniform `LayerwiseConfig` reproduces the existing config's outputs bit-for-bit.
**M2 Heterogeneous forward + extraction:** `nested.py` (layer-indexed wrappers, MLP first, attention second), `export.py` (`materialize` for per-layer), `cost.py` (per-layer params/flops/kv), `evaluate.py`. *Tests:* extend `test_nested.py`, `test_exact_float64.py`, `test_export.py` (elastic forward == materialized model in float64 for random profiles incl. kv_groups; KV-cache generation parity; heterogeneous cache, Qwen3 norms). *Success:* max abs logit diff < float64 tolerance for >=100 random profiles on 4 tiny architectures.
**M3 Sensitivity + baselines (no training):** new `search/` module or `scripts/` (sensitivity curves, random, greedy). Success: reproducible table of uniform vs random vs greedy at matched cost on the E3 checkpoint.
**M4 Hardware measurement:** new `benchmarks/` or `scripts/bench_profiles.py` (latency table, memory, noise floor, optional energy), `cost.table:` integration. Success: noise floor quantified; additivity of per-layer latency tested; measured vs FLOPs correlation reported.
**M5 Supernet training with profile sampling:** `sampling.py` (profile sampler), `train.py`/`pipeline.py` (config plumbing, YAML), `weights.py` interplay. Tests: sampler determinism and resume, coverage of narrow/wide, DDP parity. Success: no regression of uniform f025..f100 beyond a tolerance vs E3; heterogeneous profiles within X of matched-cost uniform.
**M6 Pareto/HV search + constrained variant + extraction pipeline:** search module, 3-D HV (library decision), report script. Success: HV (fixed reference) of searched front > uniform and random-profile fronts at equal evaluation budget over >=3 seeds, confirmed by end-to-end device measurement.
**M7 (optional) compile/graph deployment:** per-profile compile/capture, switch-cost benchmark. **M8 (optional, high risk):** residual-width interfaces; ViT port.

## 11. Smallest credible experiment

Hold `H`, attention and `head_dim` fixed; vary only per-layer `mlp_width_l` (SmolLM2-360M, 32 layers, 8 levels). Experiment: from the existing E3 500M checkpoint, no training, compare uniform vs random vs sensitivity-greedy vs evolutionary heterogeneous MLP profiles at matched parameter count and (once measured) matched latency, on val ppl. If heterogeneous no-train profiles are not competitive, fine-tune with profile sampling (Stage 1) before concluding. Option B (add `kv_groups_l`) next; Option C (changing residual width between blocks) only after, requiring interface projections and touching embeddings/lm_head/norms.

## 12. Benchmark / ablation matrix

Rows: baselines (a)-(h) above. Columns: val ppl (WikiText-2, C4, FineWeb-Edu), later downstream accuracy, params, FLOPs, KV bytes, latency (prefill/decode, bs1/bs>1), peak memory, energy, HV (fixed ref), search cost, compile overhead, switch cost. Ablations: sandwich vs naive for profile sampling; with/without KD; distilling from full vs neighbour; profile-sampling distribution; correlated vs iid random profiles; number of levels K; importance ordering act_weight vs taylor; MLP-only vs MLP+attention heterogeneity; proxy size vs rank correlation with end-to-end; HV reference-point sensitivity; extracted vs extracted+fine-tuned vs independent training.

## 13. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Shape mismatch (cache, GQA, alignment, strided views) | silent wrong output / crashes | float64 equivalence tests vs materialized model; heterogeneous-cache test early (M2) |
| Under-trained profiles (random profiles rarely seen) | proxy ranks mislead | correlated sampling, coverage metrics, rank-correlation check proxy vs fine-tuned |
| Weight-sharing interference / full-model drift (already +6.5% observed) | quality loss at full width | KD from frozen full model, per-config weights (R4b), lower lr, monitor drift |
| Hardware measurement noise | false Pareto ordering | noise floor, medians, fixed clocks, no ranking below noise |
| FLOPs/latency mismatch | wrong objective | optimise measured latency; report FLOPs only as secondary |
| Compilation / graph overhead, recompiles | deployment cost | per-profile static graphs, small supported set, measure compile and memory |
| Search-space size / evaluation budget | search noise | proxy cache, seeded fronts, budget-matched comparisons |
| HV pitfalls (reference point, normalisation, many objectives) | misleading claims | fixed anchors and reference, <=3 objectives, report reference |
| Qwen3/other-model ordering regressions | limited generality | SmolLM2 first; Taylor ordering ablation before other models |
| Single seed | claims unsupported | >=3 seeds for headline results (owner decision) |
| Compute | schedule | see estimate |

**Compute estimate [P]:** reference point is the 500M-token E3 run at 17.6 h on 2xA40 [R]. Stage 1 supernet continuation at ~0.5-1B tokens is therefore roughly 18-36 h per seed on that setup, times seeds; proxy evaluation is cheap (minutes per few hundred profiles if eval set is ~1M tokens, unmeasured); device measurement scales with front size. DAS-6 rules apply (GPU on 205-208 only, never `--mem`, submit and report job IDs).

## 14. Claims requiring evidence

"Heterogeneous widths beat uniform" (matched measured latency, >=3 seeds, with noise bars); "HV search beats random/greedy/evolutionary-scalar" (budget-matched, fixed reference); "lower FLOPs gives lower latency" (measure; do not assume); "extracted profile equals supernet sub-model" (float64 tests); "novelty" (literature review); "switching is cheap" (measured on target); results transfer to other models or ViTs (not yet shown).

## 15. Acceptance criteria (summary)

M1-M2: all legacy tests pass, equivalence tests pass in float64; M4: noise floor and additivity reported; M6: searched front dominates or matches uniform at equal measured latency for at least one budget with non-overlapping noise bars, across 3 seeds; deployment: extracted profile within tolerance of supernet metric; compile/capture times and switch cost documented.

## 16. Open questions for the project owner

1. Scope: LM-only first (ViT code was deleted), or restore a ViT track for a SHARD comparison?
2. Target hardware and measurement access (A40 only, or Jetson/other; energy counters available)?
3. Is the first deliverable MLP-only heterogeneity, or MLP+attention?
4. Objective set and device budgets (latency/memory limits); is 3-D HV acceptable (new dependency)?
5. Seeds: move from single seed to >=3 for headline results?
6. Is runtime switching among a handful of profiles enough, or is arbitrary runtime width change required?
7. Merge the uncommitted R4b weighting work first, or branch from the current commit?
8. Compute allowance per phase (GPU-hours) on DAS-6?
9. Is a full literature review a prerequisite before positioning claims?

## 16b. Addendum: OpenReview paper `ymUOPsbxLi` (NOT READ)

Status: **unread.** `openreview.net/pdf?id=ymUOPsbxLi` and the forum page both return a bot-verification page to the fetch tool, and a web search for the ID returned nothing. I did not try to bypass the check. So this plan makes **no claim** about what that paper adds, and no related-work cell cites it.
To close this: paste the PDF text or abstract (or drop the PDF in the repo), then add it to the M0 literature table. Questions to answer from it: does it parameterise subnetworks as subspaces of shared weights; does it use hypervolume for training or for selection; does it vary widths per layer; does it measure hardware cost; which models/datasets; is there code. Check each against section 3's matrix and revise the "genuinely different" row accordingly.
Because of this, the plan's only evidence about hypervolume use is internal: the repo's uncommitted R4b work uses HV sensitivities (`cost.py:hv_widths`, `weights.py`) to *reweight the loss across the shared-weight configs during training* [F]. That is HV as a training signal over nested sub-spaces, which is distinct from the plan's HV as a *search/selection metric over per-layer profiles*. Combining both is a candidate contribution, but novelty needs the literature review (M0) and cannot be asserted from this session.

### arXiv 2509.17874 "Deep Hierarchical Learning with Nested Subspace Networks for LLMs" (Rauba, van der Schaar) [X: read via HTML fetch, first ~100k of ~165k chars; not confirmed to be the same paper as the OpenReview ID]
- **Method [X]:** each linear layer is reparameterised as low-rank factors A (R x d_in), B (d_out x R); rank-r sub-model uses the first r rows of A / columns of B, so lower ranks are strict subspaces ("nested subspace property"). Pre-trained models are adapted by SVD-initialising A,B for the **MLP linear layers**, then fine-tuning. Loss: uncertainty-weighted (Kendall-style learned log-variances) cross-entropy at an anchor rank and at a sampled rank. Cost metric: FLOPs (break-even rank d_in*d_out/(d_in+d_out)). Models: Pythia-2.8B, GPT-Neo-2.7B, Gemma-2B, Qwen2-0.5B on an NLI task; ~5 pt accuracy loss for 50% fewer FLOPs on Pythia. Code: github.com/pauliusrauba/nested-subspace-networks (not inspected).
- **What it adds for us:**
  1. A *different nesting axis*: rank (low-rank subspace of each weight matrix) vs our neuron/head-prefix nesting. Rank nesting changes the parameterisation (extra factorisation, FLOPs savings only above the break-even rank), whereas ours keeps dense weights and slices; ablation candidate "SVD-rank nesting of MLP vs importance-ordered channel prefix at matched measured latency".
  2. Learned uncertainty weighting of anchor/variant losses is a direct comparator for our R4b per-config weights (fixed max-weight vs HV-sensitivity weights). Add "Kendall uncertainty weights" as a weighting baseline in E6.
  3. Anchor-rank-plus-variant sampling resembles our sandwich sampling; their claim that curriculum sampling beats uniform is a sampler ablation for M5.
  4. The authors state ranks are **uniform across layers** and list layer-specific adaptive compute as future work, which supports the per-layer-width gap (consistent with the matrix in section 3), but is not evidence of novelty.
- **What it does NOT contain (in the portion read):** no hypervolume or Pareto-front objective, no per-layer allocation, no measured latency/energy (FLOPs only), no search. It also claims to be "the first" on its own five-criterion table; treat that as the authors' claim, not ours to repeat.
- **Insight for the "sub-space with hypervolume" framing:** nested sub-spaces of shared weights are an existing line of work (this paper, plus slimmable/MatFormer/SortedNet-style references seen only as bibliography keys); hypervolume-driven weighting or selection over those sub-spaces is the part this paper does not do. Whether that combination is new still requires the M0 literature review, so it can be described as a *candidate* differentiator, not a novelty claim.

Where to save: the user asked for this plan to be saved under `docs/` for later review without disturbing current work. Plan mode only allows writing this plan file, so the copy (for example `docs/PLAN_heterogeneous_widths.md`, new file, no existing file touched) is left until plan mode is exited and the user approves.

## 17. Verification of this plan once approved

M2: run `pytest tests/test_nested.py tests/test_exact_float64.py tests/test_export.py` on CPU (and DAS-6 CPU nodes 201-203 per cluster rules) for heterogeneous profiles; M4: noise-floor repeat runs; M6: end-to-end front measured on target device. No step should wait more than 10 minutes on cluster jobs; submit and report job IDs.
