#!/usr/bin/env bash
# Shared shell helpers. Source this file; do not execute it.
#
# Environment (all optional):
#   ELASTOFORMER_ROOT   repository root (default: parent of scripts/)
#   SCRATCH_ROOT        heavy artifacts: caches, data, models, runs, logs
#                       (default: /var/scratch/skalra/elastoformer_moo)
#   ELASTOFORMER_ENV    conda env name or venv directory (default: prune_llm)
#   ELASTOFORMER_ALLOWED_GPU_NODES / ELASTOFORMER_ALLOWED_CPU_NODES
#                       node-name regexes, matched against the whole name
#                       (defaults: node20[5-8] and node20[1-3]; the naming is unverified)

: "${ELASTOFORMER_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${SCRATCH_ROOT:=/var/scratch/skalra/elastoformer_moo}"
export ELASTOFORMER_ROOT SCRATCH_ROOT
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_cache}"
# pylibs is an overlay (transformers>=4.51 and friends) that shadows the older packages in the base
# environment without modifying it. It must come before the environment's site-packages.
export PYTHONPATH="$ELASTOFORMER_ROOT/src:$SCRATCH_ROOT/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# keep caches off the small home quota
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit "${2:-1}"; }

# Hugging Face token for gated models. Taken from the environment, else from ~/.hf_token (owner-only file
# outside the repository; scripts/sync_to_das6.sh copies it to the cluster). Never read from a tracked file:
# a token committed to the repository would also be published with the paper's artifact.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
  _mode="$(stat -c %a "$HOME/.hf_token" 2>/dev/null || true)"
  case "$_mode" in
    600|400|"") ;;
    *) log "WARNING: ~/.hf_token has mode $_mode and may be readable by other users; run: chmod 600 ~/.hf_token" ;;
  esac
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.hf_token")"
  export HF_TOKEN
  unset _mode
fi

# Name of the node this process runs on. Slurm sets SLURMD_NODENAME; NODE_OVERRIDE exists
# only so the guard can be unit-tested.
current_node() { printf '%s' "${NODE_OVERRIDE:-${SLURMD_NODENAME:-$(hostname -s)}}"; }

# require_node_class gpu|cpu : exit 3 unless this node is on the allowed list for the class.
require_node_class() {
  local kind="$1" node regex
  node="$(current_node)"
  case "$kind" in
    gpu) regex="${ELASTOFORMER_ALLOWED_GPU_NODES:-node20[5-8]}" ;;
    cpu) regex="${ELASTOFORMER_ALLOWED_CPU_NODES:-node20[1-3]}" ;;
    *) die "require_node_class: kind must be gpu or cpu, got '$kind'" 2 ;;
  esac
  if [[ ! "$node" =~ ^(${regex})$ ]]; then
    die "refusing to run $kind work on node '$node' (allowed: ^(${regex})\$)" 3
  fi
  log "node '$node' is allowed for $kind work"
}

# load_cuda_module : every GPU job on DAS-6 must load the CUDA toolkit module. Without it torch
# cannot load libcuda and silently runs on CPU. Override with ELASTOFORMER_CUDA_MODULE.
load_cuda_module() {
  local mod="${ELASTOFORMER_CUDA_MODULE:-cuda12.1/toolkit/12.1}"
  if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  type module >/dev/null 2>&1 || die "'module' is not available; cannot load $mod" 4
  module load "$mod" || die "module load $mod failed" 4
  log "loaded module $mod"
}

# init_conda : make `conda activate` available in a non-login shell. Looks for conda.sh at
# $ELASTOFORMER_CONDA_SH, then the usual DAS-6 and home locations.
init_conda() {
  # `conda activate` needs the shell function that conda.sh defines; a conda binary on PATH is not enough
  [[ "$(type -t conda 2>/dev/null)" == function ]] && return 0
  local sh
  for sh in "${ELASTOFORMER_CONDA_SH:-}" "/var/scratch/${USER:-$(id -un)}/minimonda3/etc/profile.d/conda.sh" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$sh" && -f "$sh" ]]; then
      # shellcheck disable=SC1090
      source "$sh"
      return 0
    fi
  done
  return 1
}

# activate_env [gpu] : activate $ELASTOFORMER_ENV, which may be a venv directory, a conda prefix
# directory (as on DAS-6: /var/scratch/skalra/prune_llm) or a conda env name.
# With "gpu", the CUDA module is loaded first.
activate_env() {
  local env="${ELASTOFORMER_ENV:-prune_llm}" candidate had_u=0
  [[ "${1:-}" == gpu ]] && load_cuda_module
  # conda's activation scripts read unset variables, so nounset must be off while they run
  case $- in *u*) had_u=1; set +u ;; esac
  for candidate in "$env" "$HOME/$env" "$HOME/venvs/$env" "$HOME/.virtualenvs/$env" "/var/scratch/${USER:-$(id -un)}/$env"; do
    if [[ -f "$candidate/bin/activate" ]]; then
      # shellcheck disable=SC1090,SC1091
      source "$candidate/bin/activate"
      [[ $had_u == 1 ]] && set -u
      log "activated venv $candidate"
      check_python_env
      return 0
    elif [[ -d "$candidate/conda-meta" ]]; then
      if init_conda && conda activate "$candidate"; then
        [[ $had_u == 1 ]] && set -u
        log "activated conda prefix $candidate"
        check_python_env
        return 0
      fi
    fi
  done
  if init_conda && conda activate "$env" 2>/dev/null; then
    [[ $had_u == 1 ]] && set -u
    log "activated conda env $env"
    check_python_env
    return 0
  fi
  [[ $had_u == 1 ]] && set -u
  die "environment '$env' not found as a venv, conda prefix or conda env (set ELASTOFORMER_ENV; conda.sh: ELASTOFORMER_CONDA_SH)" 4
}

check_python_env() {
  python - <<'PY' || die "the active environment cannot import torch and transformers" 4
import sys
import torch, transformers
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}", file=sys.stderr)
PY
}

# stage_data <src_dir> <dest_dir> <file>... : copy files from shared storage (NFS on DAS-6) to node-local
# storage (a local NVMe at /local on the GPU nodes) and set STAGED_DIR. Returns 1 when staging is not possible
# (missing file, destination not writable, not enough free space); the caller then keeps using the originals.
# The staged copy is removed when the script exits, including on SIGTERM from Slurm (KEEP_STAGE=1 keeps it).
# A staged file is reused only if its size matches the source.
stage_data() {
  local src="$1" dest="$2" f need=0 avail start="$SECONDS"
  shift 2
  for f in "$@"; do
    [[ -f "$src/$f" ]] || { log "stage: missing $src/$f"; return 1; }
    need=$(( need + $(stat -c %s "$src/$f") ))
  done
  mkdir -p "$dest" 2>/dev/null || { log "stage: cannot create $dest"; return 1; }
  avail="$(df -B1 --output=avail "$dest" 2>/dev/null | tail -1 | tr -d ' ')"
  if [[ -n "$avail" ]] && (( avail < need + need / 10 + 1073741824 )); then
    log "stage: only $(( avail / 1048576 )) MiB free at $dest, need $(( need / 1048576 )) MiB plus margin"
    rmdir "$dest" 2>/dev/null
    return 1
  fi
  for f in "$@"; do
    if [[ ! -f "$dest/$f" || "$(stat -c %s "$dest/$f")" != "$(stat -c %s "$src/$f")" ]]; then
      cp "$src/$f" "$dest/$f.part" && mv "$dest/$f.part" "$dest/$f" || { log "stage: copying $f failed"; return 1; }
    fi
  done
  STAGED_DIR="$dest"
  log "staged $(( need / 1048576 )) MiB to $STAGED_DIR in $(( SECONDS - start )) s"
  if [[ "${KEEP_STAGE:-0}" != 1 ]]; then
    trap 'rm -rf "$STAGED_DIR"' EXIT
    trap 'exit 143' TERM
  fi
  return 0
}

ensure_scratch() {
  mkdir -p "$SCRATCH_ROOT"/{hf_cache,data,models,runs,logs,pylibs,pip_cache} || die "cannot create $SCRATCH_ROOT" 5
}
