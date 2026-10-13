#!/usr/bin/env bash
# Compile flash-attn from source into the pylibs overlay (CPU-only work: nvcc needs no GPU).
#
# Why a source build: the prebuilt flash-attn wheel in the base environment needs glibc 2.32 and the DAS-6
# nodes have 2.28, so it cannot even be imported. Compiling on the cluster links against the local glibc.
# The result goes to $SCRATCH_ROOT/pylibs, which sits before the base environment on PYTHONPATH, so it
# replaces the unusable copy without modifying the shared environment.
#
#   TIME_LIMIT=04:00:00 scripts/submit.sh cpu submit_build_flash_attn.slurm
#
# Environment (all optional):
#   FLASH_ATTN_VERSION   default 2.8.3 (the version already in the base environment)
#   FLASH_ATTN_CUDA_ARCHS default 80. sm_80 code runs on sm_86 (A40, A10) because a cubin built for
#                        compute capability X.y runs on X.z with z >= y; one arch keeps the build short
#   MAX_JOBS             parallel nvcc processes (default 12; nvcc is memory-hungry, the node has ~125 GB)
#   ELASTOFORMER_CUDA_MODULE  default cuda12.6/toolkit/12.6, matching torch's CUDA 12.6 build
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib.sh

require_node_class cpu
ensure_scratch
export ELASTOFORMER_CUDA_MODULE="${ELASTOFORMER_CUDA_MODULE:-cuda12.6/toolkit/12.6}"
activate_env gpu   # "gpu" here only means "load the CUDA toolkit module"; no GPU is used

LIBS="$SCRATCH_ROOT/pylibs"
command -v nvcc >/dev/null || die "nvcc not found after loading $ELASTOFORMER_CUDA_MODULE" 4
log "$(nvcc --version | tail -1)"

# Without ninja, PyTorch's extension builder silently falls back to compiling one file at a time (the
# login node has /usr/bin/ninja but the compute nodes do not; a first attempt compiled 24 of ~72 kernels in
# 35 minutes on a single core). So require it, installing the ninja wheel into the overlay if needed.
if ! command -v ninja >/dev/null; then
  log "ninja not found on this node; installing the ninja wheel into the overlay"
  python -m pip install --quiet --target "$LIBS" --upgrade ninja
  export PATH="$LIBS/bin:$PATH"
fi
command -v ninja >/dev/null || die "ninja is required for a parallel build and could not be installed" 4
log "gcc $(gcc -dumpfullversion), ninja $(ninja --version)"
python - <<'PY'
import torch
print(f"building against torch {torch.__version__} (CUDA {torch.version.cuda})")
PY

VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
# FORCE_BUILD stops setup.py from downloading the prebuilt wheel, which is the glibc-incompatible one.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-12}"
export CC=gcc CXX=g++
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"; mkdir -p "$TMPDIR"

log "building flash-attn==$VERSION (arch $FLASH_ATTN_CUDA_ARCHS, MAX_JOBS=$MAX_JOBS) into $LIBS"
start=$(date +%s)
python -m pip install --target "$LIBS" --upgrade --no-deps --no-build-isolation "flash-attn==$VERSION"
log "build finished in $(( $(date +%s) - start )) s"

# The metadata shim from setup_pylibs.sh is only for a flash_attn that cannot load; it would now hide the real one.
rm -rf "$LIBS/flash_attn-0.0.0.dist-info"

log "import check (a GPU is not needed to import; the real test is scripts/bench_attention.py on a GPU node)"
python - <<'PY' || log "WARNING: import failed on this CPU node; verify on a GPU node before trusting the build"
import flash_attn
print("flash_attn", flash_attn.__version__, "from", flash_attn.__file__)
assert "pylibs" in flash_attn.__file__, "the overlay copy is not the one being imported"
PY
ls "$LIBS" | grep -i flash_attn || true
