# Plan: SenSys 2027 submission — nested elastic small language models on Jetson Orin Nano

Written for: the authors (planning document, not for reviewers).

## 1. Decisions (current)

1. **Target:** SenSys 2027, this cycle, regular research track, full paper (12 pages, ACM `acmart` sigconf, double-blind). Timelines are not tracked in this plan.
2. **Scope:** everything in the journal plan (`research/ELASTOFORMER_EXECUTION_PLAN.md`) for **small language models**. No ViT training. ViT results are an optional extension at the end of the paper, only if space remains.
3. **Prior work:** the ElastoFormer SEC '25 paper (Kalra and Sapra, DOI 10.1145/3769102.3770612) and its released per-level checkpoints remain available for reference.
4. **Relationship to prior work:** a new work that builds on ElastoFormer. The prior paper is referred to in the **third person** throughout, as the CFP requires for self-citations.

## 2. Positioning rules

1. **Third person for the prior work.** "Kalra and Sapra [ElastoFormer] proposed …", "the original ElastoFormer design …". Use "we" only for the new work.
2. **Third person is necessary, not sufficient.** The CFP requires original, unpublished work. The introduction must state exactly what is new relative to SEC '25: the SLM setting, the nested training recipe, and the edge measurements.
3. **Do not hide the prior paper.** Cite it in the introduction, related work, and wherever the design is inherited (nesting by prefix, importance ordering).
4. **Figures and text from SEC '25.** Do not reuse its figures or Table 1 without checking ACM author-rights terms. Cite published numbers explicitly as prior work, or reproduce them.
5. **No author-identifying material.** `anonymous` option in `acmart`; no names or affiliations; anonymous artifact link (for example anonymous.4open.science); no GitHub user name, no cluster paths; PDF metadata stripped.
6. **Generative AI.** AI used to conduct the research (analysis, experiment code, data processing) is described in the methods section. AI used only for writing needs no disclosure.
7. **Ethics.** No human-participant research is planned.

## 3. Core claim and research questions

Candidate claim (to be tested, not assumed):

> A single pretrained decoder-only language model, with its weights partitioned into nested prefixes, can expose several operating configurations whose quality, memory, token latency and energy differ measurably, can be switched at runtime without loading a second checkpoint, and reaches a better worst-case quality across configurations than naive prefix slicing of the pretrained model.

Research questions:

- **RQ1 — Feasibility:** which pretrained SLM can be converted into nested configurations that run on the Orin Nano 8 GB, and at what quality cost?
- **RQ2 — Training:** does joint nested training with distillation, sandwich sampling and importance-ordered initialization recover quality at the smallest configurations without losing the full model?
- **RQ3 — Trade-off control:** do learned per-configuration weights (hypervolume-driven) and the monotonicity penalty improve the set of configurations, compared with fixed weights?
- **RQ4 — Systems:** do configurations on shared weights give lower switching cost and comparable or lower memory than separately stored checkpoints, measured on the device?

## 4. Model candidates (architecture facts to verify before use)

**E0 was run on 2026-10-13** (`scripts/check_models.py`, results in `results/e0_model_facts_2026-10-13.json`). 18 of 24 registry entries were fetched and matched their registered `model_type`; Llama 3.2 and Gemma are gated (no `HF_TOKEN` set) and SmolLM3 is not recognized by transformers 4.51.1, so those stay unverified. The table below keeps the original selection rationale; the measured facts supersede any number from memory. Qwen2.5-3B reports license `other`, which needs reading before publication.

| Model | Role | Why | Check before use |
|---|---|---|---|
| SmolLM2-360M | Development model | Fast iteration; small enough for many runs | heads, KV heads, intermediate size, tied embeddings, license |
| SmolLM2-1.7B | Primary candidate | Standard decoder-only design, straightforward slicing | same fields |
| Llama-3.2-1B | Primary candidate (alternative) | Widely used; 8 KV heads gives attention elasticity by KV group | license terms (Llama community license) |
| Qwen2.5-1.5B | Stretch | Has few KV groups, so attention elasticity is coarse; checks generality | KV head count, bias terms in QKV |

Selection rule: choose the primary model after the architecture check, preferring the one with the most KV groups and the simplest license for publication. Confirm SmolLM3 sizes before listing it; do not assume a sub-3B checkpoint exists.

## 5. Elastic design for decoder-only SLMs

### 5.1 Elastic dimensions (first implementation)

- **MLP intermediate width (SwiGLU):** keep a prefix of the intermediate channels in gate and up projections, and the matching columns of the down projection. Primary elastic dimension.
- **Attention in whole KV groups:** keep a prefix of KV heads together with their query heads. This keeps head dimension and RoPE unchanged and shrinks the KV cache per configuration.
- **Layer depth (optional):** only in the extension stage; not in the first implementation.
- **Hidden / residual width (RMSNorm, tied embeddings):** fixed in the first implementation. Deferred to the extension stage.

Why not head-dimension pruning: prior ViT work showed that pruning head dimension shrinks every head at once; for RoPE models, keeping head dimension intact avoids re-deriving rotary frequencies per configuration.

Widths are aligned to multiples of 256 channels so that quantized prefix views (K-quant blocks of 256) stay valid.

### 5.2 Importance ordering and permutation

- Compute activation-aware importance per KV group and per MLP channel on a calibration set (for example a few thousand tokens of general text).
- Apply the permutation once to the stored weights so that each configuration is a prefix. Apply it consistently across Q/K/V, output projection, gate/up/down, and any normalization affecting those channels.
- Validate the permutation: the full model's outputs must be unchanged (up to numerical tolerance) before any training.
- This keeps the nesting property: the SEC '25 ordering idea is kept, but applied once to a language model rather than per pruning round.

### 5.3 Interaction checks (must be tested, not assumed)

- Grouped-query attention: a KV group removal changes the number of query heads per configuration consistently.
- Tied embeddings: the embedding is fixed in the first implementation, so tying does not change configurations.
- SwiGLU: gate and up must be permuted identically.
- KV cache: memory per configuration is computed from KV groups times context length; report it separately from weights.
- Tensor-parallel or fused kernels: not used; single-device inference.

### 5.4 Quantization (treated as a separate axis)

- Stage A: post-training quantization of each configuration's weights (4-bit and 8-bit), measured against full precision.
- Stage B: quantization-aware nesting (optional): train the nested prefixes under quantization noise.
- Report quantization and elasticity separately, and report the combination. Do not assume low-bit weights reduce latency on the Orin Nano without measuring it.

## 6. Training recipe (staged)

1. **Load and inspect** the pretrained model; confirm the architecture facts in section 4.
2. **Baselines:** full model, naive prefix slicing (no training), and independently pruned models at matched sizes (for example SliceGPT or LLM-Pruner outputs, or a comparable structured pruner) as the "bag of models" reference.
3. **Zero-shot quality** before fine-tuning, per configuration.
4. **Importance ordering and permutation** (section 5.2) with the equivalence check.
5. **Diagnosis:** per-configuration perplexity and downstream accuracy; identify which configurations degrade most.
6. **Joint nested training with distillation:**
   - Sandwich sampling each step: largest, smallest, one or two random intermediate configurations.
   - Loss: next-token cross-entropy per sampled configuration, plus a distillation term matching the frozen full model's token distributions (KL on logits; hidden-state alignment on shared prefixes as an optional term).
   - No frozen shared parameters; distillation is the anchor for the largest configuration.
   - Optional low-rank per-configuration deltas (section 7.4), with their storage reported.
7. **Monotonicity penalty** (section 7.2) as an ablation.
8. **Learned configuration weights** (section 7.1) as an ablation against fixed weights.
9. **Quantization** stage A, then stage B if time allows (section 5.4).
10. **Validate on the device** (section 8).

Parameter-efficient training: LoRA is used only as the per-configuration delta in section 7.4, not to train the whole elastic model, so that the shared-weight claim is not weakened.

## 7. Multi-objective components

Distinction kept throughout: multi-objective optimization over training losses (sections 7.1–7.3) is separate from multi-objective selection over deployment configurations (section 8.4).

### 7.1 Hypervolume-driven configuration weights (main MOO candidate)

- Adapted from GMOOAR in the thesis of Chen (2026, Ch. 3). Per-configuration loss weights are learned in a first-order bilevel loop:
  - inner step: update shared weights with the current weights;
  - outer step: update the weights to increase the hypervolume of the set {(measured cost, validation loss)} per configuration.
- Cost axis: measured latency or memory on the Orin Nano (not FLOPs).
- Choosing configuration widths is a post-hoc search over the trained model, because every prefix is valid.

### 7.2 Monotonicity penalty (cheap ablation)

- Penalize [L_{i+1} − L_i]₊ for nested configurations i ⊂ i+1, so that a larger configuration is not worse than a smaller one. Adapted from the multi-forward ordering loss in PaMaL and LORPMAN (thesis Ch. 4).

### 7.3 Conflict-aware updates (gated)

- Compute per-configuration gradients on shared parameter groups and their cosine similarity.
- Add PCGrad, CAGrad or MGDA-style updates only if the measured conflict is substantial. Otherwise report the measurement and omit the method.

### 7.4 Per-configuration low-rank deltas (optional)

- From LORPMAN and Pareto Merging: shared nested base plus a small per-configuration low-rank adapter on attention output or MLP down projection.
- Must be reported as explicit configuration-specific storage (for example "under 1% of base weights per configuration"), so that the single-model claim stays honest.
- Hot-swappable at switch time; the switch cost is measured in section 8.

### 7.5 Methods considered and not used in the main method

- COSMOS-style input conditioning and hypernetworks (PHN): conditioning changes computation per configuration and hypernetworks are larger than the base model. Both conflict with zero-copy nested inference. Listed in related work only.
- Evolutionary search: reserved for choosing configuration widths, not for training.

## 8. Jetson Orin Nano 8 GB: runtime, measurement and controller

### 8.1 Hardware facts to verify

- Shared LPDDR5 system memory (no discrete VRAM). Verify capacity, bandwidth in each power mode, and the JetPack version.
- Memory budget for the model: total 8 GB minus OS, CUDA context, and other resident processes. The usable budget is measured, not assumed.

### 8.2 Runtime path

1. PyTorch reference measurements (correctness and a first latency estimate).
2. Patched llama.cpp (or equivalent GGUF runtime) that builds each configuration's graph from prefix views of one GGUF file, so no tensor is copied or reloaded at switch time. Verify the runtime's CUDA-on-Jetson support and its handling of quantized block strides before committing.
3. Baseline: separately stored GGUF checkpoints per configuration (the "bag of models" baseline).

### 8.3 Metrics

- Time to first token, prefill latency, decode tokens per second, peak memory, model storage, energy per generated token (tegrastats and an external or on-board power measurement; method stated in the paper).
- Prompt-length sensitivity and KV cache growth.
- Switching overhead: prefix view selection versus reloading a checkpoint.
- Theoretical estimates, simulator or proxy numbers, and on-device measurements are reported separately.

### 8.4 Adaptive controller

- Input: available memory, latency deadline, energy budget, or a quality requirement.
- Output: one configuration from the same shared-weight model, chosen from a table of measured cost per configuration.
- Compare against a controller over independently stored checkpoints, using the same policy.

## 9. Evaluation

### 9.1 Quality

- Perplexity on held-out text (WikiText-2 and a web-text set such as C4; exact sets to fix in the protocol).
- Downstream zero-shot tasks chosen for the model family (for example ARC-Easy/Challenge, HellaSwag, PIQA, WinoGrande). The exact list is fixed before running, and is reported with the evaluation harness version.
- Instruction-following: optional; only if the primary model is an instruction-tuned variant.

### 9.2 Nested-specific metrics

- Quality per configuration; quality relative to the full model.
- Worst-case degradation across configurations; average quality across configurations.
- Pareto front and hypervolume over (measured cost, quality).
- Shared storage and per-configuration overhead.
- Quality under a simulated runtime change (switching between configurations in sequence).

### 9.3 Baselines (all compared under matched compute and evaluation settings)

- Full pretrained model.
- Naive prefix slicing (no training).
- Importance-ordered slicing (section 5.2), no training.
- Independently pruned models at matched sizes.
- Independently stored quantized checkpoints (bag of models) for the systems comparison.
- Published nested or elastic LLM methods (for example MatFormer, Flextron, LLaMaFlex, Nemotron-Elastic): cite and compare where code or reported numbers are available; do not reproduce them unless the code is available and runs.

## 10. Experiment list

Costs are estimates to confirm before committing.

| ID | Experiment | Purpose | Est. cost (2×A40 / Orin Nano) |
|---|---|---|---|
| E0 | Architecture and license check of each candidate; choose primary model | RQ1 | 0.5 day |
| E1 | Full model, naive slicing, importance ordering: quality per configuration, no training | RQ1 baseline | 1 day |
| E2 | Permutation equivalence check; diagnosis of per-configuration degradation | RQ1 | 0.5 day |
| E3 | Joint nested training with distillation vs naive recipe at matched compute (SmolLM2-360M first) | RQ2 | 1–2 days on 360M |
| E4 | Ablations: distillation on/off, sandwich on/off, importance ordering on/off | RQ2 | 2–3 days on 360M |
| E5 | Primary model (1–1.7B) joint training under the same recipe | RQ2 main result | 1.5–3 days (estimate) |
| E6 | Monotonicity penalty and learned configuration weights vs fixed | RQ3 | 1–2 days |
| E7 | Gradient-conflict measurement; conflict-aware update only if justified | RQ3 | 0.5 day measurement |
| E8 | Quantization stage A (4-bit, 8-bit) per configuration | Quantization axis | 1 day |
| E9 | Jetson Orin Nano: TTFT, decode tok/s, peak memory, energy, prompt sweep, per configuration | RQ4 main systems table | 2–3 days |
| E10 | Switching cost: prefix view selection vs checkpoint reload | RQ4 claim | 1 day |
| E11 | Adaptive controller under changing budgets, vs independently stored checkpoints | Systems demonstration | 2 days |
| E12 | Per-configuration low-rank deltas: quality gain vs storage overhead | Optional | 1–2 days |

Compute estimates from the journal plan: about 8 h for SmolLM2-360M over 500M tokens on 2×A40; about 1.5–3 days for a 1–1.7B model over 0.5–1B tokens with ZeRO-2 and gradient checkpointing. These are estimates, not measurements.

**Priority order if compute or device time runs short:** E0, E1, E2, E3, E9, E10, E5, E4, E6, E8, E11, E7, E12. The minimum publishable core is E0–E3, E9 and E10 on the smaller model, plus E5 for the primary model.

## 11. Decision gates (technical, not dated)

- **Gate A (after E3):** joint nested training improves the smallest configuration over the naive recipe at matched compute, with the largest configuration within a stated tolerance of the full model. If not, the training contribution is weakened and the paper focuses on the systems and diagnosis content.
- **Gate B (after E9 and E10):** the nested runtime shows measurable switching cost advantage over checkpoint reload, and the per-configuration cost table is stable across runs. If not, the systems claim is reduced to the measured numbers.
- **Gate C (after E5):** the primary model reproduces the 360M trend. If not, the paper is restricted to the development model and says so.

## 12. ViT extension (optional, end of paper)

- Only if space remains after the SLM results.
- Evaluation only: no ViT training. Use the released ElastoFormer per-level checkpoints to show that the same nested evaluation protocol (per-configuration quality, worst-case, switching) applies to vision transformers.
- Reported as an extension or compatibility result, in one short section after the SLM conclusions.
- If the SLM results need the space, drop this section entirely.

## 13. Risks and open questions

1. **Architecture fit:** the chosen SLM may have few KV groups or tied and biased projections that complicate slicing. Gate: E0.
2. **Runtime fit:** the patched llama.cpp path may not give zero-copy views for every quantized type. Fall back to reporting the PyTorch and GGUF baselines, and state the limitation.
3. **Memory:** the 1–1.7B primary model may leave too little headroom on 8 GB shared memory once the KV cache and OS are counted. Measure before committing to a context length.
4. **Novelty vs prior work:** nested training of LMs is an active area (Flextron, MatFormer, LLaMaFlex, Nemotron-Elastic). The related-work search (pending) must confirm the specific combination of prefix nesting, distillation, learned configuration weights and measured edge switching is not already published.
5. **Overlap with SEC '25:** the CFP requires original work. The SLM setting is new; the diagnosis and design reuse must be disclosed and cited.
6. **Literature search status:** not yet done for this scope. The Awesome-LLMs-Pruning list and the Chen et al. surveys are the starting points. Every citation in the final paper must be verified.
7. **Generative AI disclosure:** if any AI tool is used in experiment code or analysis, it is described in the methods section.

## 14. Next steps

1. Literature and novelty search for nested and elastic LLMs and the MOO-for-LLM literature (pending).
2. E0: architecture and license check for the candidate models, and Orin Nano specifications for the chosen JetPack and power mode.
3. E1 and E2 on SmolLM2-360M.
4. Fill the LaTeX draft in `paper/sensys2027/` (already rewritten to the SLM framing; every result is still a `\TODO` placeholder).

## 15. Files

- `research/SENSYS_2027_PLAN.md`: this plan.
- `research/ELASTOFORMER_EXECUTION_PLAN.md`: the journal plan this scope is drawn from.
- `paper/sensys2027/`: LaTeX source for Overleaf, in the SLM framing; results are placeholders.
- `src/elastoformer_moo/`, `scripts/`, `tests/`: the implementation (see `README.md`).
