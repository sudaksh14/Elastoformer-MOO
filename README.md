# Elastoformer-MOO

Nested shared-weight elastic inference for decoder-only small language models.

License: Apache-2.0 (see `LICENSE`).

One pretrained model, several operating points. Each smaller configuration is a **prefix** of a larger
one, so all configurations share the same weights and switching changes which slice is used, not which
checkpoint is loaded. A configuration keeps the first `kv_groups` key-value groups of attention and the
first `mlp_width` channels of every MLP.

This repository builds on ElastoFormer (SEC '25), which applied the nested idea to vision transformers
with frozen, bottom-up growth. Here all configurations are trained **jointly**.

## Status

Implemented and tested on CPU: nested forward, importance ordering, joint training with distillation,
perplexity evaluation, the pipeline stages, the cluster node guards. **Not implemented:** quantization,
the Jetson runtime and measurements, hypervolume-driven configuration weights, downstream benchmark
evaluation, multi-GPU training, depth elasticity. No experiment has been run at scale. See
`IMPLEMENTATION_STATUS.md` and `EXPERIMENT_STATUS.md` for the exact state.

## How it works

- `nested.py` patches `forward` on each decoder block's attention and MLP so they slice the shared
  weights with views. `state_dict` keys, `save_pretrained` output and parameter names are unchanged.
  Attention keeps whole KV groups, so head dimension, RoPE and the KV-cache layout are untouched.
- `importance.py` and `permute.py` reorder weights once (most important first) and verify the permuted
  model computes the same logits.
- `train.py` samples the largest, the smallest and random intermediate configurations each step,
  sums their cross-entropy plus a distillation term from a frozen copy of the full model, and updates
  the shared weights. An optional monotonicity penalty discourages a larger configuration from being
  worse than a smaller one.

## Supported models

Any Hugging Face model whose `model_type` is `llama`, `mistral`, `qwen2` or `qwen3`: Llama 3.x,
TinyLlama, MobileLLaMA, SmolLM and SmolLM2, OpenLLaMA, Qwen2/2.5, Qwen3. Gemma, Phi and SmolLM3 raise
`UnsupportedArchitectureError` with the reason. `configs/models.yaml` lists the registry; facts fetched
from the hub on 2026-10-13 are in `results/e0_model_facts_2026-10-13.json`. Gated models (Llama 3.2,
Gemma) need `HF_TOKEN` and are unverified until it is set.

## Quick start

```bash
# environment: any with torch>=2.4 and transformers>=4.51 (tested: torch 2.6, transformers 4.51.1)
./run_local_checks.sh                       # syntax, shell syntax, CPU unit tests

python scripts/check_models.py              # E0: fetch and compare model configs (network)
python scripts/prepare_tokens.py --tokenizer llama3.2-1b --inputs corpus/*.txt --out $SCRATCH_ROOT/data/train.bin
python scripts/permute_model.py --model llama3.2-1b --calib $SCRATCH_ROOT/data/calib.bin --out $SCRATCH_ROOT/models/llama3.2-1b-permuted
python scripts/train_nested.py --config configs/experiments/llama3.2-1b_nested.yaml --set train.steps=50
python scripts/eval_ppl.py --model $SCRATCH_ROOT/models/llama3.2-1b-permuted --tokens $SCRATCH_ROOT/data/val.bin
```

```python
from transformers import AutoModelForCausalLM
from elastoformer_moo import make_elastic, make_grid

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
ctrl = make_elastic(model)
configs = make_grid(ctrl.arch, fractions=(0.25, 0.5, 0.75, 1.0))
with ctrl.use(configs[1]):                  # run at the second operating point
    logits = model(input_ids=ids).logits
print(ctrl.active_parameters(configs[1]))   # parameters used at this point
```

Importance ordering is not guaranteed to help. In a no-training smoke check on about 6k held-out tokens
(`results/real_model_check_*.json`), it cut the 25% configuration's perplexity from 3e7 to 5e3 on SmolLM2-135M
and from 8e5 to 5e3 on Qwen2.5-0.5B, but **made Qwen3-0.6B worse** (4.6e5 to 8.7e7). The scoring heuristic is
untuned and the calibration set tiny; treat these as a reason to compare orderings, not as results.

## DAS-6

GPU work runs only on nodes 205-208 and CPU-only work on nodes 201-204. Submit through the wrapper, which
excludes every other node, and the job scripts re-check the node before doing anything:

```bash
scripts/submit.sh cpu submit_cpu_tests.slurm
scripts/submit.sh gpu submit_gpu_tests.slurm
MODEL=llama3.2-1b TIME_LIMIT=12:00:00 scripts/submit.sh gpu submit_experiments.slurm train
```

Verified on the cluster on 2026-10-13: node names `node201`-`node208`; partitions `defq` (GPU) and `fatq` (CPU);
`--gres=gpu:1` (or `gpu:A40:1`); GPU jobs load `cuda12.1/toolkit/12.1`; no `--mem` (nodes report 1 MB). Heavy files
go to `$SCRATCH_ROOT` (default `/var/scratch/skalra/elastoformer_moo`), never into this repository.

One-time setup, from Git Bash on the laptop (the ssh key is passphrase-protected, so load it into an agent first):

```bash
eval "$(ssh-agent -a ~/.ssh/agent.sock)" && ssh-add ~/.ssh/id_rsa    # once per session
scripts/sync_to_das6.sh                                              # copies the repo to /home/skalra/Elastoformer-MOO
ssh skalra@fs2.das6.science.uva.nl 'cd Elastoformer-MOO && bash scripts/setup_pylibs.sh'
```

`setup_pylibs.sh` builds an overlay (transformers 4.51.1) that shadows the cluster environment's transformers 4.28
without modifying that shared environment. The environment's prebuilt `flash_attn` cannot load on these nodes (glibc 2.28),
so it is compiled from source into the overlay on a CPU node:

```bash
ssh skalra@fs2.das6.science.uva.nl 'cd Elastoformer-MOO && TIME_LIMIT=02:00:00 scripts/submit.sh cpu submit_build_flash_attn.slurm'
```

Flash attention is requested explicitly (`ATTN`, default `flash_attention_2`) and never silently replaced by `sdpa`.
On SmolLM2-360M it is numerically equivalent to `sdpa` and not faster (`EXPERIMENT_STATUS.md`).

Multi-GPU training (data parallel, one process per GPU; the job must own the GPUs, so request `gpu:A40:2` on node205):

```bash
MODEL=smollm2-360m TIME_LIMIT=02:00:00 scripts/submit.sh cpu submit_experiments.slurm train-data   # tokenize the corpus once
NPROC=2 MODEL=smollm2-360m TIME_LIMIT=05:00:00 GPU_GRES=gpu:A40:2 scripts/submit.sh gpu submit_experiments.slurm train
```

Training data is copied to the node-local NVMe at job start (`STAGE_LOCAL=0` disables it) and removed at exit.

Running the baseline experiments for a model (one stage per job; each needs an explicit `TIME_LIMIT`, no `--mem`):

```bash
MODEL=smollm2-360m TIME_LIMIT=00:20:00 scripts/submit.sh cpu submit_experiments.slurm download
MODEL=smollm2-360m TIME_LIMIT=00:40:00 scripts/submit.sh cpu submit_experiments.slurm data
MODEL=smollm2-360m TIME_LIMIT=00:30:00 GPU_GRES=gpu:A40:1 scripts/submit.sh gpu submit_experiments.slurm e1e2
```

## Hugging Face token (for gated models such as Llama 3.2)

Free. Create a Hugging Face account, open the model page (for example `meta-llama/Llama-3.2-1B`), accept the license
there (Meta approves access; how long that takes varies), then create a *read* token at
huggingface.co/settings/tokens. Export it as `HF_TOKEN` in the shell that downloads the model; never commit it or paste it into chat. SmolLM, Qwen,
TinyLlama and MobileLLaMA need no token.

Where: the token is only needed by whatever fetches a gated model. On DAS-6 that is the `download` stage, which
downloads into the cache on `/var/scratch`; later jobs read the cache. Locally it is only needed to run
`check_models.py` on gated entries.

```bash
# Linux, DAS-6 or Git Bash: current shell only; `read -rs` keeps the token off the screen and out of history
read -rs HF_TOKEN && export HF_TOKEN
```
```bat
rem Windows cmd: current window only
set HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```
```powershell
# Windows PowerShell: current window only
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"
```

For a Slurm job, `sbatch` passes the submitting shell's environment through (`--export=ALL`), so export it in the shell
that runs `scripts/submit.sh`. Do not store the token under `/var/scratch` (readable by other users) or put it on a command line
(visible in `ps`); `scripts/download_model.py` reads it only from the environment.

## Layout

```
src/elastoformer_moo/   arch, config, nested, importance, permute, losses, sampling, data,
                        evaluate, train, pipeline, registry, cluster
scripts/                check_models, prepare_tokens, permute_model, train_nested, eval_ppl,
                        gpu_smoke, submit.sh, inspect_das6.sh, lib.sh, node_policy.py
configs/                models.yaml, experiments/*.yaml
tests/                  CPU unit tests; test_gpu.py is skipped without CUDA
results/                small recorded results (E0 model facts)
research/               plans and notes
paper/sensys2027/       LaTeX source for the SenSys 2027 draft
```

## Known limits

- Transformers internals: the elastic attention reuses Hugging Face's attention functions and RoPE. It is
  tested on transformers 4.51.1; other versions may change the attention call signature.
- `Wo[:, :k]` and `Wd[:, :k]` are strided views. Whether PyTorch copies them on each call is not measured.
- Training is single-device. 3B-class models need more than one A40 for full-parameter training.
- Resuming restores weights, optimizer, scheduler and sampler state, but the data stream is re-seeded from
  `(seed, step)`, so a resumed run is not bit-identical to an uninterrupted one.
