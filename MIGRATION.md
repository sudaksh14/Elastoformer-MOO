# Migration from the original ElastoFormer repository

Written for: the maintainers. Remove or anonymize this file before any anonymous artifact release
(it names the old repository URL).

## What happened

The original vision-transformer repository (`Elastoformer/`, 30 tracked files, 4 commits) was retired
on 2026-10-13 and this repository was created in its place. Before deletion:

- the working tree was clean, and
- `git ls-remote origin refs/heads/main` returned `df028db2604c8f8ca2bf4d87af78445e95d30097`, the same
  commit as the local `HEAD`, so GitHub already holds the complete history.

The old code stays available at `https://github.com/sudaksh14/Elastoformer`, commit `df028db`.

## Why not carry the old code over

The old code is specific to vision transformers and CNNs and to a different training method:

| Old file(s) | What it did | Why it is not ported |
|---|---|---|
| `main.py`, `trainer.py` | Prune with `torch_pruning`, fine-tune the core, regrow larger levels with frozen shared weights | Frozen growth is the behaviour this work replaces with joint nested training |
| `models/elastoformer.py`, `models/vit_hf.py`, `models/vit_timm.py`, `models/deit.py`, `models/swin_transformer.py`, `models/mlp_mixer.py`, `models/convnext.py` | ViT-family models; the elastic ViT is written against `transformers==4.28` | Vision only; the pinned `transformers` is too old for Llama 3.x and Qwen3 |
| `elastic_cnn.py`, `models/Resnet.py`, `models/VGG.py`, `utils/cnn_prune_utils.py` | CNN support | CNNs are out of scope |
| `utils/prune_utils.py` (about 1,700 lines) | Index bookkeeping to rebuild each level from pruning metadata | Replaced by one permutation plus prefix views; no per-level rebuild exists any more |
| `utils/partial_freezing.py` | Gradient hooks that freeze shared weights | No weights are frozen in joint training |
| `datasets.py`, `augment.py`, `sampler.py` | ImageNet and CIFAR loading, augmentation, repeated-augmentation sampler | Image-specific |
| `compare_tensor.py`, `run.sh`, `images/` | Checkpoint comparison, run recipe, README figures | Specific to the ViT experiments |

## What was reused

No source file was copied. The following ideas carry over, and the new code is written from scratch:

- **Importance-first structured selection.** ElastoFormer ranks channels by L1 norm before pruning.
  Here the ranking is activation-aware and computed once (`importance.py`, `permute.py`) so every
  configuration is a prefix.
- **Nested weight sharing.** Smaller operating points are subtensors of larger ones. Here this is
  implemented with views over the shared tensors (`nested.py`) instead of copied per-level models.
- **Training defaults.** AdamW, linear warm-up then cosine decay, no weight decay on norms and biases,
  gradient clipping at 1.0 (`train.py`).
- **Repository hygiene.** `.gitignore` patterns for caches and checkpoints.

## Not carried over, on purpose

- The ViT evaluation extension planned in `research/SENSYS_2027_PLAN.md` (section 12) would need the old
  ViT model code and its `transformers==4.28` environment. It is not implemented here. If it is wanted,
  run it from the old repository at commit `df028db` rather than porting it.

## Inventory of the retired repository (30 tracked files)

```
.gitignore  README.md  augment.py  compare_tensor.py  datasets.py  elastic_cnn.py  main.py
requirements.txt  run.sh  sampler.py  trainer.py
images/{Resnet_latency_nano,TX_freezing,accuracy_plot,latency_nano,latency_orin,scheme}.png
models/{Resnet,VGG,convnext,deit,elastoformer,mlp_mixer,swin_transformer,vit_hf,vit_timm}.py
utils/{cnn_prune_utils,partial_freezing,prune_utils,train_utils}.py
```

History: `5e70772` Initial commit (2025-12-10), `d7215bb` and `2587960` README and citation (2026-01-14),
`df028db` updated dependencies (2026-09-28).
