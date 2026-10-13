#!/usr/bin/env bash
# Build the `pylibs` overlay on DAS-6: a directory of newer packages that shadows the older ones in the
# base environment without modifying it. Run it on the login node (it downloads small wheels).
#
#   scripts/setup_pylibs.sh
#
# Why: the base environment (prune_llm) has transformers 4.28, which predates Llama 3.x and Qwen3 and
# lacks the attention API this code reuses. The overlay adds transformers 4.51.1 + tokenizers 0.21.1 and
# pytest. torch, numpy and everything else still come from the base environment.
#
# Environment: SCRATCH_ROOT, ELASTOFORMER_BASE_PYTHON (default /var/scratch/skalra/prune_llm/bin/python)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PY="${ELASTOFORMER_BASE_PYTHON:-/var/scratch/skalra/prune_llm/bin/python}"
[[ -x "$PY" ]] || die "base python not found: $PY (set ELASTOFORMER_BASE_PYTHON)" 4
ensure_scratch
LIBS="$SCRATCH_ROOT/pylibs"

log "installing into $LIBS using $PY"
# --no-deps: pip ignores the packages already installed in the base environment when it targets a
# directory, so resolving dependencies would pull a second copy of torch.
"$PY" -m pip install --quiet --target "$LIBS" --upgrade --no-deps "transformers==4.51.1" "tokenizers==0.21.1"
"$PY" -m pip install --quiet --target "$LIBS" --upgrade pytest
# pyarrow reads the WikiText-2 parquet files (scripts/prepare_eval_data.py); it has no required dependencies
"$PY" -m pip install --quiet --target "$LIBS" --upgrade --no-deps pyarrow

# The base environment may ship a flash_attn build whose compiled extension needs a newer glibc than the
# cluster nodes have (DAS-6: glibc 2.28; flash_attn 2.8.3 needs 2.32). transformers 4.51 imports flash_attn
# whenever CUDA is visible and the package metadata says it is installed, so on a GPU node it crashes while
# importing any model (CPU nodes never notice). Shadowing the *metadata* with version 0.0.0 makes
# transformers conclude flash-attention is unavailable. The shared environment is not modified.
# A real build in the overlay (scripts/build_flash_attn.sh) supersedes the shim, so never create one next to it.
if compgen -G "$LIBS/flash_attn_2_cuda*.so" >/dev/null; then
  log "overlay contains a compiled flash_attn; no metadata shim"
  rm -rf "$LIBS/flash_attn-0.0.0.dist-info"
elif ! PYTHONPATH="" "$PY" -c "import flash_attn" >/dev/null 2>&1; then
  if PYTHONPATH="" "$PY" -c "import importlib.metadata as m; m.version('flash_attn')" >/dev/null 2>&1; then
    log "flash_attn is installed but cannot be imported; shadowing its metadata in the overlay"
    shim="$LIBS/flash_attn-0.0.0.dist-info"
    mkdir -p "$shim"
    printf 'Metadata-Version: 2.1\nName: flash_attn\nVersion: 0.0.0\nSummary: placeholder so transformers does not import a flash_attn build that cannot load on this cluster\n' > "$shim/METADATA"
  fi
fi

log "verifying the overlay is what Python imports"
PYTHONPATH="$LIBS" "$PY" - <<'PY'
import sys, tokenizers, transformers, torch
print(f"transformers {transformers.__version__} from {transformers.__file__}")
print(f"tokenizers   {tokenizers.__version__} from {tokenizers.__file__}")
print(f"torch        {torch.__version__} (from the base environment)")
assert transformers.__version__ == "4.51.1" and "pylibs" in transformers.__file__, "overlay is not being used"
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # the API the elastic attention needs
print("ALL_ATTENTION_FUNCTIONS import ok")
import importlib.metadata as md
try:
    print("flash_attn metadata version seen by transformers:", md.version("flash_attn"))
except md.PackageNotFoundError:
    print("flash_attn: not installed")
PY
log "pylibs ready"
