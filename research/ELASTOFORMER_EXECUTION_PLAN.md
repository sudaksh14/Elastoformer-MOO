# Plan: ElastoFormer journal extension — research diagnosis, MOO, elastic SLMs on Jetson Orin Nano

## Context

The user wants a journal extension of ElastoFormer (Kalra & Sapra, SEC '25, DOI 10.1145/3769102.3770612). The extension should keep the paper's identity: one pretrained model holding several nested, shared-weight operating modes for adaptive edge inference. It should also fix the large accuracy loss at CR = 0.8, add multi-objective optimization (MOO), and move the method to decoder-only SLMs (< 3B parameters) deployed on a Jetson Orin Nano 8GB.

The deliverables are five research documents plus a figure plan. No code changes are part of this task.

**Constraints the user gave:**
- Compute: 2× A40 (48 GB each).
- Framing: the SLM work leads the paper.
- Existing per-level ViT checkpoints (surfdrive) can be used.
- A physical Orin Nano 8GB is available for measurements.
- No CNNs anywhere in scope.

Workspace at the time of writing: `Elastoformer/` (the original ViT code, a separate git repo, since retired; its code citations below link to commit `df028db` on GitHub) and `related papers/`. The second folder contains:

- **MOO-pruning works.** MOSP (NeurIPS'25), MOONSHOT (arXiv 2604.13287) and *Multi-Objective Structured Pruning of LLMs for Latency…* (arXiv 2607.22583). All three produce **independent** pruned models, not nested ones. They are the closest MOO-pruning works, not nested competitors.
- **MOO methodology (added by the user).**
  - COSMOS (arXiv 2103.13392): preference-conditioned single network that approximates the whole Pareto front in one run.
  - Peitz & Hotegni survey (arXiv 2412.01566).
  - Chen et al. gradient-MOO survey (arXiv 2501.10945), with the companion list Baijiong-Lin/Awesome-Multi-Objective-Deep-Learning.
  - **Weiyu Chen PhD thesis** (HKUST, Jan 2026, *Efficient MOO for Deep Learning*):
    - Ch. 3 GMOOAR: adaptive reference vectors through bilevel optimization, with a hypervolume (HV) quality term.
    - Ch. 4 LORPMAN: a shared main network plus per-objective low-rank matrices, orthogonal regularization, a "freeze epoch" after which only the low-rank parts train, and the PaMaL multi-forward *ordering* regularizer.
    - Ch. 5 Pareto Merging: a preference-independent base plus a preference-dependent low-rank tensor.
    - Ch. 6 MOSP: a common core support plus task-specific supports.

## Verified findings so far (from reading the code and the paper)

These go into the research plan as the "evidence" section. Items marked *verify* are checked against the checkpoints in P0.

1. **The pipeline is Compress → Grow, and the core is the *smallest* model.**
   - Compress: `torch_pruning.BasePruner` with L1 group magnitude, a uniform per-layer ratio, `prune_head_dims=True`, `prune_num_heads=False`. The residual/hidden dimension is pruned too; only the classifier is ignored ([main.py:282-308](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/main.py#L282-L308)).
   - The core is fine-tuned alone ([main.py:481](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/main.py#L481), `fine_tuner_core`).
   - Each larger level is a **freshly built model** (`create_vit_general`). Weights are copied in at merged/sorted original indices (`update_vit_weights_global`, [prune_utils.py:906](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/utils/prune_utils.py#L906)). The previous level's weights are frozen by gradient hooks (`freeze_partial_weights`, [prune_utils.py:1196](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/utils/prune_utils.py#L1196)). Only the new weights are trained ([main.py:523-540](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/main.py#L523-L540)).
2. **What CR means.**
   - In the code, `--pruning_ratio` is the fraction of channels removed per prunable layer, scheduled *linearly* over steps. Level k keeps 1 − CR·(N+1−k)/N of the width.
   - At CR = 0.8 the core keeps 20% of the width and about 3.6% of the FLOPs (0.60 / 16.86 G). This matches paper Table 1 under a quadratic FLOP model.
   - Paper Eq. (1) instead defines CR = FLOP_min/FLOP_B with a geometric per-round ratio. **Neither part of that matches the code.** This inconsistency must be fixed in the journal version.
3. **The degradation is not confined to the core.**
   - CR 0.8: L1 51.19%, L2 64.14%, …, L6 (full size) 74.65% against a base of 81.74%. The full-capacity model loses 7.1 points.
   - The full-size model loses 5.3 points at CR 0.6 and 1.6 points at CR 0.4.
   - Every larger level inherits a frozen subnet that was optimized for the core alone. **The method sacrifices the maximum model by construction.**
4. **Gradient conflict cannot cause the current result.** The code never trains configurations jointly. Conflicts only appear once joint training is introduced, which is exactly what the extension proposes. They will be measured then (E0.4, E2.x).
5. **Parameters that are not actually nested (code-level, verify on checkpoints).**
   - `cls_token` and `position_embeddings` are not copied into descendants. `update_vit_weights_global` handles only Linear, LayerNorm and Conv2d, so these start from random `post_init` values and are trained per level.
   - LayerNorm **bias** is not frozen: `freeze_layernorm_params` masks only the weight.
   - The classifier is skipped entirely, so it is trainable per level.
   - Each level is saved as a separate state dict ([main.py:562](https://github.com/sudaksh14/Elastoformer/blob/df028db2604c8f8ca2bf4d87af78445e95d30097/main.py#L562)).
   - Consequence: a smaller level cannot be recovered *exactly* from the largest model. The "single set of parameters" claim needs an audit.
6. **Core under-recovery is likely.**
   - The CR = 0.8 core is about 154-d with 12 heads of about 13 dims, because the head count is fixed and head dims are pruned.
   - It gets no distillation and is fine-tuned for 50 epochs. DeiT-Ti (192-d, 3×64 heads) needs 300 epochs plus distillation to reach about 72%.
   - `get_num_heads` re-derives the head count from {6, 8, 10, 12} by divisibility. That could re-partition heads between levels (*verify*).
7. **Hyperparameters differ between the paper and the code.** The paper says lr 3e-3 and batch size 128. `run.sh` uses lr 5e-5 and batch size 512. Descendants use weight decay 0. We need to find out which settings produced Table 1.
8. **Switching uses non-contiguous index gathers, not prefix slices.** That is consistent with the reported 50.4 ms switch on Orin. For SLMs the nesting must be made **contiguous by a one-time permutation**, so that each configuration is a zero-copy view.

**Implication:** the root weakness is the frozen bottom-up growth, plus an untuned, undistilled extreme core and leaks in the nesting. The natural fix is joint nested training, done after a one-time importance permutation, with distillation from the original model. That is also where MOO and gradient conflict become meaningful. The novelty risk is high, because Flextron, MatFormer/Gemma 3n, LLaMaFlex, Nemotron Elastic and SortedNet all do joint nested training of LMs. The contribution has to be positioned carefully (see the novelty section).

## Execution steps (after approval)

### Step 1 — Literature and code verification (WebSearch / WebFetch, read-only)

- Walk through the categories of `github.com/liyunqianggyn/Awesome-LLMs-Pruning`. Classify each relevant entry as independent-model, multi-config without shared weights, or nested/shared/adaptive. Follow the papers and repositories of the most relevant ones.
- Verify each of these: paper, year, venue, official repo, whether the repo still exists, license, and the key numbers.
  - **Nested and elastic:** Slimmable/US-Nets, OFA, BigNAS, HAT, DynaBERT, MatFormer, Gemma 3n (MatFormer deployment), Flextron, LLaMaFlex, Nemotron Elastic, SortedNet/Sorted LLaMA, Matryoshka Representation Learning, MatQuant/any-precision LLM.
  - **Compression:** LLM-Pruner, SliceGPT, Wanda, SparseGPT, ShortGPT, Minitron, Sheared-LLaMA.
  - **Dynamic depth:** LayerSkip, early-exit LMs, Mixture-of-Depths.
  - **Quantization:** GPTQ, AWQ, bitsandbytes, llama.cpp/GGUF.
  - **MOO:** MGDA, PCGrad, CAGrad, GradNorm, uncertainty weighting, Nash-MTL, plus MOSP / MOONSHOT / 2607.22583.
  - **Pareto-set learning** (from the new PDFs): EPO, PMTL, PHN (Pareto HyperNetworks), COSMOS, YOTO, PaMaL, GMOOAR, LORPMAN, Pareto Merging.
- Walk `github.com/Baijiong-Lin/Awesome-Multi-Objective-Deep-Learning` as a second curated list, focusing on its NAS, compression and LLM sections.
- **Targeted novelty search:** "Pareto" / "multi-objective" / "hypervolume" combined with "nested", "Matryoshka", "elastic", "once-for-all", "supernet" or "slimmable". Check whether any work already treats nested-subnetwork training as Pareto-set learning, or adapts configuration weights by HV.
- Pull HF `config.json` and license for each SLM candidate:
  - SmolLM2-360M and SmolLM2-1.7B
  - SmolLM3 (confirm which sizes exist)
  - Qwen2.5-0.5B, 1.5B and 3B (check the 3B license)
  - Qwen3-0.6B and 1.7B
  - Llama-3.2-1B and 3B
  - Gemma-3-1B
  - Phi models under 3B
  
  For each, record heads, KV heads, head_dim, intermediate size, layer count, tied embeddings and license.
- **Jetson:** pull NVIDIA's specs for Orin Nano 8GB: shared LPDDR5, bandwidth in standard and "Super" (JetPack 6.2) modes, and power modes. Check llama.cpp CUDA-on-Jetson support, ggml view and stride rules for quantized blocks (K-quant super-block of 256), MLC-LLM, ONNX Runtime GenAI, and TensorRT-LLM/Edge-LLM support on Orin.

### Step 2 — Write the deliverables

New folder: `C:\Users\skalra\Workspace\Elastoformer-MOO\research\`. No existing files are touched.

| File | Key content |
|---|---|
| `ELASTOFORMER_JOURNAL_RESEARCH_PLAN.md` | Summary of the paper and code; claim→code map (table of file:line refs); the findings 1–8 above with evidence; CR = 0.8 analysis; MOO formulation; training recipe; SLM architecture; Jetson strategy; novelty; risks; journal narrative |
| `ELASTOFORMER_RELATED_WORKS.md` | Comparison table: shared weights / nested / adaptive / LM / on-device / code status (official / third-party / unavailable / unverified) / relevance / role (baseline, dependency, competitor); dedicated Awesome-LLMs-Pruning section |
| `ELASTOFORMER_EXPERIMENT_MATRIX.md` | P0–P4 experiments with hypothesis, baseline, metric, compute on 2×A40, dependencies, success criteria, continue/abandon rules |
| `ELASTOFORMER_SLM_IMPLEMENTATION_PLAN.md` | Model choice, elastic dimensions, permutation/conversion and validation, training and distillation, quantization, Jetson runtime, dependencies, staged plan |
| `ELASTOFORMER_JOURNAL_OUTLINE.md` | Paper structure, contributions, how it goes beyond SEC'25, and a figure plan (architecture, training pipeline, Pareto/MOO process, runtime controller) |

## Technical design to be written into the documents

**MOO formulation (recommended).** Use the smallest method that addresses the demonstrated weakness:
- min over θ of [ L_max(θ), mean_i L_i(θ), max_i L_i(θ) ], with configurations i as nested prefixes of θ.
- Treat it as a *constrained* problem: L_max ≤ L_max^anchor + ε. This protects the full model, which is exactly what the SEC'25 method lost.
- Deployment budgets (memory, latency, energy) select the *configuration set* through measured lookup tables. They are not part of the training loss.
- Keep two things apart: MOO over training losses, versus multi-objective search over deployment configurations (runtime selection, Eq. (10) of the paper, extended to a Pareto set).

Method ladder, each step taken only if the previous one leaves a measured gap:
1. Fixed weights with a sandwich rule (max + min + random).
2. Uncertainty or GradNorm weighting.
3. Conflict-aware updates on the shared parameter groups: PCGrad, CAGrad or MGDA-UB, applied per layer. Only if E0.4/E2.1 show negative cosine similarity.
4. A Lagrangian max-anchor constraint.

Evolutionary search is reserved for choosing configuration widths, not for training.

**What changes after the new PDFs.**

- **(a) Formal framing.** A nested elastic model *is* finite Pareto-set learning: K solutions on the quality–cost front. The parameter-sharing structure is the distinctive part:
  - PaMaL uses a linear combination of base networks.
  - LORPMAN uses a shared base plus low-rank parts.
  - COSMOS uses input conditioning.
  - ElastoFormer uses **nested prefixes**, where every smaller solution is a subtensor of every larger one.
  
  This gives the journal paper a principled MOO lens, using the taxonomy of Chen et al. 2501.10945, instead of "we applied MGDA". It becomes the main candidate contribution, pending the novelty search in Step 1.
- **(b) HV-adaptive configuration weights.** This is GMOOAR transferred to nested models. The reference vectors become per-configuration loss weights, learned in a first-order bilevel loop:
  - Inner step: train the shared weights.
  - Outer step: update the weights to maximize the hypervolume of {(measured Jetson latency or memory, validation loss) per configuration}.
  
  This directly answers the SEC'25 future-work item ("automated search for the number of DNs and the CRs"). Choosing configuration *widths* stays a cheap post-hoc search over the trained model, because every prefix at a multiple of 256 is valid.
- **(c) Monotonicity / ordering regularizer**, adapted from the PaMaL/LORPMAN multi-forward loss: penalize [L_{i+1} − L_i]_+ for nested configurations i ⊂ i+1. A larger configuration should never be worse. This is a cheap P1 ablation.
- **(d) Optional per-configuration low-rank deltas**, from LORPMAN and Pareto Merging: shared nested base plus a tiny per-configuration LoRA, norm or head. It absorbs residual conflict without copying base weights and is hot-swappable at switch time.
  - It must be reported as explicit configuration-specific storage, for example "< 1% per configuration", so the single-model claim stays honest.
  - It also formalizes the existing ViT leaks (per-level classifier, LayerNorm bias, cls/pos) as a deliberate design choice, instead of an accident.
  - It is the answer to the brief's §5.4 LoRA question.
- **(e) LORPMAN's "freeze epoch" mirrors ElastoFormer's freezing**, but in the *opposite* order: LORPMAN trains the shared part jointly first and freezes later; ElastoFormer freezes from the start. That contrast motivates the ablation "joint-then-freeze versus freeze-from-start".
- **(f) MOSP's "common core support"** is conceptually ElastoFormer's core. MOSP is unstructured and yields independent sparse models, so it gives no Jetson speedup and no nesting. It goes into the related-works table as a positioning contrast. Its dual-ADMM core selection is noted as a P4 option for picking the structured core under several calibration objectives.
- **Rejected for the main method:** COSMOS-style input conditioning and hypernetworks (PHN). Conditioning changes the computation for every configuration, and hypernetworks are larger than the base model. Neither fits zero-copy nested inference on an 8 GB device. They appear in the related-works table only.

**Training recipe.** Staged:
1. Importance permutation: activation-aware ordering of heads/KV-groups and MLP channels, plus calibration. This reuses the Compress step as the *ordering*, so Elastoformer's identity is kept.
2. A short recovery of the full model.
3. Joint nested training with distillation from the frozen original model (logit KL, plus optional hidden-state alignment on shared prefixes). Configuration sampling uses a core-first curriculum, which turns the "Grow" insight into a schedule rather than freezing.
4. Optional max-config recovery phase.

The ablation order is defined in the experiment matrix.

**SLM elastic dimensions.** First implementation:
- MLP intermediate width: SwiGLU gate/up rows plus down_proj columns.
- Attention in **whole KV groups** (each KV head with its query heads). This keeps head_dim, so RoPE is unaffected, and shrinks the KV cache.
- Optional layer depth.

The hidden and residual width (RMSNorm, tied embeddings) stays fixed; that is deferred to P4. Head-dim pruning is explicitly avoided, motivated by finding 6. Widths are aligned to multiples of 256 so that quantized prefix views are valid.

**Candidate models.**
- Dev: SmolLM2-360M.
- Primary: Llama-3.2-1B (8 KV groups) or SmolLM2-1.7B (MHA, easiest).
- Stretch: Qwen2.5-1.5B, which has only 2 KV groups and therefore coarse attention elasticity.

The final choice depends on the Step 1 verification.

**Compute estimates.** These are labeled as estimates in the documents:
- SmolLM2-360M joint training over 500M tokens: about 8 h on 2×A40.
- A 1–1.7B model over 0.5–1B tokens: about 1.5–3 days with ZeRO-2 plus gradient checkpointing.
- Full ImageNet ViT-B re-runs are out of budget. The ViT work uses existing checkpoints for diagnosis, and DeiT-S on ImageNet-100 or 10% of ImageNet for counterfactuals.

**Jetson.**
- Memory budget: shared 8 GB minus OS/CUDA, with about 5–6 GB usable (to be measured). The budget covers weights, quantization scales, KV cache by context length, activations and the runtime.
- Runtime path:
  1. PyTorch reference measurements.
  2. Patched llama.cpp: build each configuration's graph from `ggml_view` prefix views of one GGUF, so nothing is copied or reloaded.
  3. Compare against separate GGUF checkpoints, the "bag of models" baseline.
- Metrics: TTFT, prefill and decode tok/s, peak memory, J/token (tegrastats / INA3221), prompt-length sweep and switching overhead.
- Theory, proxy numbers and on-device measurements are reported separately.

## Priorities (summary of the matrix)

- **P0** — ViT, mostly evaluation only:
  - Reproduce Table 1 from the checkpoints.
  - Nesting audit: accuracy of Level-k *extracted from Level-6* against the stored Level-k.
  - Head and dimension audit per level.
  - Per-configuration gradient cosine probe on shared parameters.
  - Small-scale counterfactuals: frozen-grow against joint training, with and without distillation, head-dim pruning against head pruning.
- **P1** — fix the nesting leaks, add distillation from the original model, and use joint sandwich training (ViT small scale plus SmolLM2-360M). Also run the ordering regularizer and the "joint-then-freeze versus freeze-from-start" ablation.
- **P2** — main method: nested Pareto-set learning, made of:
  - a max-anchor constraint;
  - HV-adaptive configuration weights (GMOOAR-style, with the HV computed on Jetson-measured cost);
  - conflict-aware updates, only if measured conflict justifies them;
  - optional per-configuration low-rank deltas.
  
  Ablated on SmolLM2-360M, then the primary SLM. Metrics add **hypervolume** and the worst-case gap to an *independently trained or pruned* model at each size, the "bag of models" upper reference.
- **P3** — the primary SLM (1–1.7B) on Orin Nano: patched llama.cpp, zero-copy switching and an adaptive controller.
- **P4** — quantization-aware nesting at int4, depth elasticity, hidden-width elasticity.

Decision gate before committing to the full direction:
- (a) The P0 audit confirms finding 3 or 5.
- (b) On SmolLM2-360M, joint nested training beats frozen-grow and naive slicing at the smallest configuration, with the max-config loss staying within ε of the original.
- (c) The patched llama.cpp shows decode speed growing with width on Orin Nano, and switching measurably cheaper than reloading GGUF.

## Verification of the deliverables

- Every paper or repository claim in `RELATED_WORKS` carries a URL that was fetched during Step 1. Anything not fetched is labeled "unverified".
- Every code claim cites `file:line` from this workspace.
- Paper numbers are copied from Table 1–4 of `related papers/Elastoformer.pdf`.
- Facts, hypotheses, proposed methods and verified results are tagged separately.
- No existing file is modified. Only the new `research/` folder is created.
- Final chat reply follows §12 of the brief:
  1. Diagnosis Q1–6.
  2. Literature and repository highlights.
  3. Novelty assessment.
  4. Experiment priorities.
  5. Journal outline.
  6. The first experiments to run and the evidence needed before committing to the full direction.
