# Training corpora, token budgets and evaluation across the related work

Written for: the authors, to fix the training corpus, token budget and benchmarks for E3 and later experiments.

Provenance: every entry was read from the paper's own text (PDFs extracted with `pdftotext`) on 2026-10-13. Where a detail was
not found in the text I searched, the cell says **not found** rather than a guess. Numbers marked *(derived)* are my arithmetic
on stated values, not quotes.

## 1. The works in `related papers/`

"Calibration" means data used to score weights in one-shot pruning, with no weight training. "Train" means gradient training of
the model. **None of the LLM works in this folder trains on a corpus**: they prune one-shot, so their "token budget" is a
calibration set of a few hundred thousand tokens.

| Work | Kind | Models (size) | Calibration or training data | Token budget | Evaluation | Hardware |
|---|---|---|---|---|---|---|
| **ElastoFormer** (SEC '25) | nested ViT/CNN, frozen growth | ViT-B (DeiT-B), ResNet-50, VGG-16 | Train: ImageNet-1K (1,281,167 images), CIFAR-10/100 | 50 epochs per descendant network, 5 warm-up, repeated augmentation 3x; 6 networks | top-1 accuracy, FLOPs, latency (Jetson Orin, Nano), memory | 8x RTX 3090 (24 GB) |
| **MOSP** (NeurIPS '25) | multi-objective one-shot unstructured pruning | Llama-2 7B and 13B, Llama-3 8B, OPT 1.3B, 2.7B, 30B | Calibration, one disjoint set per objective: a C4 subset (the split from ALPS), GSM8K (official split), a Python-code instruction set (Alpaca style, CC-BY-4.0); a fourth objective, ChineseWebText, in the appendix | 128 segments of up to 2,048 tokens per objective, about 262k tokens *(derived)*; no training | perplexity on the C4, GSM8K and code test sets; hypervolume; time and GPU memory (Llama-2-7B: 1.36 h, 20.3 GB) | 8x A6000 (48 GB), one GPU per run, about 100 h in total |
| **MOONSHOT** (arXiv 2604.13287) | one-shot pruning, two objectives, LLMs and vision | Llama-3.2-1B, Llama-3.2-3B, Llama-2-13B-chat, Llama-3.2 Instruct; DeiT-Tiny/Small/Base; ResNet-50 | Calibration: 128 C4 samples (LLMs); 4,096 ImageNet-1k samples (vision) | no training; 3 seeds | perplexity on C4, WikiText-2, PTB; zero-shot BoolQ, PIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OpenBookQA; ImageNet-1k accuracy | A100 80 GB, L40 40 GB; 4x L40 for vision |
| **Structured MOO pruning** (arXiv 2607.22583) | two-stage structured pruning, latency-aware | Llama-2 7B, Qwen2.5 7B, Phi-3 14B, Mistral-v0.3 7B | Search: 1,024 random FineWeb-Edu samples (KL divergence); validation on WikiText-2 and a "mixed calibration" set; calibration size varied 128 to 2,048 samples | no training; search about 28 GPU-hours; 5 runs | zero-shot (lm-evaluation-harness) MMLU, HellaSwag, ARC-easy, ARC-challenge, PIQA, WinoGrande; perplexity on WikiText-2, FineWeb-Edu, C4; 100-token generation latency on A100 and a Jetson device | 3x A100 80 GB |
| **COSMOS** (arXiv 2103.13392) | preference-conditioned Pareto front | small CNNs, EfficientNet-B4 | MultiMNIST/Fashion, Adult/Compass/Default (fairness), CelebA | epochs, no tokens | hypervolume | not LLM; not relevant to corpora |
| **Chen thesis** (HKUST 2026) | MOO toolbox | Ch.6 is MOSP; Ch.3-5 use multi-task vision datasets (MultiMNIST, Census, UTKFace, CIFAR-100, CelebA, Cityscapes) and ViT-B/32 and ViT-L/14 model merging | as the chapters above | as above | hypervolume, task accuracy | as MOSP |
| **Two MOO surveys** (2412.01566, 2501.10945) | surveys | none | none | none | none | no experiments |
| **OLMES** (arXiv 2406.08446, newly added) | an evaluation standard | 15 base models, 1B to 70B (including Pythia-1B) | n/a | n/a | 10 multiple-choice tasks: ARC-Challenge, ARC-Easy, BoolQ, CommonsenseQA, HellaSwag, MMLU, OpenBookQA, PIQA, Social IQa, WinoGrande; curated 5-shot prompts; both multiple-choice (MCF) and cloze (CF) formulations, best one reported | n/a |

Common ground in this folder: **C4 is the standard LLM calibration corpus** (MOSP, MOONSHOT, and the Wanda/SparseGPT/ALPS lineage
they follow), usually **128 samples of up to 2,048 tokens**. **WikiText-2 and C4 perplexity** appear in most LLM papers.
The zero-shot set that recurs is **ARC-easy, ARC-challenge, PIQA, WinoGrande, HellaSwag**, plus BoolQ and OpenBookQA in MOONSHOT
and MMLU in the structured-pruning paper.

## 2. Supplementary: nested or prune-and-recover LLM works that do report training budgets

Not in the folder. Read from the PDFs on 2026-10-13 (the Minitron entry is the first paper, "Compact Language Models via Pruning
and Knowledge Distillation"). These are the only evidence here for how many tokens a training-based method uses.

| Work | Method | Models | Training data | Training tokens | Evaluation |
|---|---|---|---|---|---|
| **Flextron** (2406.10260) | nested elastic + routers, continued training from a pretrained model | GPT-3 2B and 8B, Llama-2 7B | "the same domain data" as pretraining; 2.62B extra tokens from the Nemotron-4 curated continued-training set for Llama-2 joint tuning | 89.9B for elastic continued training (7.54% of GPT-3's pretraining) + 1.049B router tuning; batch 256, 80,000 steps; 3 random sub-models plus the full model per step; importance sorting from 512 samples | ARC-e, LAMBADA, PIQA, WinoGrande, MMLU (5-shot), HellaSwag (10-shot) |
| **LLaMaFlex** (ICLR 2025) | nested + router, 4 budgets (25/50/75/100%) | Llama 3.1 8B | a proprietary high-quality pretraining dataset | 60.4B tokens counted over 4 budgets; batch 128, sequence 4,096, 28,800 iterations, lr 4e-5, which is about 15.1B unique tokens x 4 budgets *(derived)* | ARC-e, LAMBADA, PIQA, WinoGrande, MMLU (5-shot), HellaSwag (10-shot) |
| **Nemotron Elastic** (2511.16664) | nested + router + frozen-teacher distillation | Nemotron Nano V2 12B parent; 9B and 6B nested | the Nemotron Nano V2 compression data blend (sources not named in the text I read) | about 110B: 65B at sequence 8,192 (batch 1,536) + 45B at sequence 49,152 (batch 512); 1,024 calibration samples at 8,192 | MATH-500, AIME-2024/2025, GPQA, LiveCodeBench v5, MMLU-Pro |
| **MatFormer** (2310.07707) | nested FFN, trained from scratch | decoder LMs 78M to 850M, ViT-B/L | **not found** in the text I searched | **not found** (trained on up to 256 TPU-v4 cores for 3 days) | validation loss, 1-shot downstream tasks, ImageNet-1K |
| **Sheared-LLaMA** (2310.06694) | structured pruning to a target architecture + continued pre-training | LLaMA-2 7B to 1.3B and 2.7B | RedPajama, with dynamic batch loading by domain | 50B tokens in total for pruning and continued pre-training (5% of OpenLLaMA's pretraining budget) | downstream tasks (list not extracted) |
| **Minitron** (2407.14679) | structured pruning + distillation | Nemotron-4 15B to 8B and 4B | Nemotron-4 curated data | less than 3% of the original data; up to 40x fewer tokens than training from scratch; 1.8B tokens for the lightweight-retraining ablation | MMLU is reported; the full benchmark list was not extracted |
| **LLM-Pruner** (2305.11627) | structured pruning + LoRA recovery | LLaMA-7B, Vicuna-7B, ChatGLM-6B | importance from N = 10 samples; a small LoRA post-training corpus (about 50 MB, 3 hours on one GPU); dataset name and epochs **not found** in my search | tiny | WikiText-2, PTB perplexity; zero-shot BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OpenBookQA |
| **SliceGPT** (ICLR 2024) | PCA slicing, optional recovery fine-tuning | OPT 125M to 66B, LLaMA-2 7B/13B/70B, Phi-2 | calibration: WikiText-2 train or Alpaca train | 1,024 samples of 2,048 tokens (about 2.1M tokens *(derived)*); H100 | WikiText-2 perplexity; zero-shot PIQA, WinoGrande, HellaSwag, ARC-e, ARC-c |

SmolLM2-360M (our development model) was pretrained on 4T tokens on 128 H100s (its model card). Its public pretraining corpus
`HuggingFaceTB/smollm-corpus` is ungated, licensed ODC-By, and contains `fineweb-edu-dedup` (234 parquet files, 550 GB),
`cosmopedia-v2` and `python-edu`.

## 3. What the evidence supports for E3

1. **No work in the folder gives a training-token budget for an LLM.** The relevant evidence is Table 2, and it is
   large: 15B to 110B tokens, which is roughly 0.4% to 8% of the base model's pretraining.
2. **What we can afford is far smaller.** The attention benchmark measured a forward+backward step of about 0.19 to 0.25 s per
   4 sequences of 1,024 tokens on an A40 (bf16). With the sandwich rule (largest, smallest, one random configuration) plus a teacher
   forward, that is roughly 0.7 s per 4,096 unique tokens, about 5-6k unique tokens per second per GPU *(derived estimate, not an
   end-to-end measurement)*. Two A40s: about 11k tokens/s at best, so 200M tokens is about 5 hours, 500M about 13 hours, 1B about
   25 hours. Even 1B tokens is about 0.025% of SmolLM2-360M's 4T pretraining, and about 15 times (1B against LLaMaFlex's 15.1B unique tokens) to several hundred times (0.2B against Nemotron Elastic's 110B) below the budgets in Table 2.
   Any E3 result is therefore a **small-budget result** and must be reported as such.
3. **Corpus.** The pretraining-like data used by the training-based works matches SmolLM2's own public corpus. Using
   `smollm-corpus/fineweb-edu-dedup` keeps the training distribution close to the base model's, and FineWeb-Edu is already used
   in the structured-pruning paper. Calibration for importance scoring stays on C4 (the pruning lineage's standard), and evaluation
   uses WikiText-2 and C4 validation, which are disjoint from the training corpus.
4. **Benchmarks.** Perplexity on WikiText-2 and C4 (done in E1/E2). For downstream accuracy, the recurring set is ARC-easy,
   ARC-challenge, PIQA, WinoGrande and HellaSwag, with BoolQ and OpenBookQA from MOONSHOT; run under the OLMES protocol
   (5-shot, cloze formulation for small models, because multiple-choice formatting is near random for small models, per OLMES).
   MMLU is likely near chance at 360M and would need the cloze formulation.

Proposed staging (the authors may override): a 50M-token pilot to measure real throughput and check that the loss moves, then a
500M-token main run, then 1 to 2B tokens if the trend holds. These numbers are my proposal from the table above, not a result.

Sources: the PDFs in `related papers/` and arXiv 2310.07707, 2310.06694, 2305.11627, 2401.15024, 2406.10260, 2407.14679,
2511.16664, plus the ICLR 2025 LLaMaFlex paper.
