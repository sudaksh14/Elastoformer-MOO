#!/usr/bin/env bash
# GPU tests for DAS-6 nodes 205-208. Exits 3 on any other node before touching the GPU.
# There is deliberately no override: GPU work must never run on nodes 201-204.
#
#   ./run_gpu_tests.sh [pytest args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/lib.sh

require_node_class gpu
ensure_scratch
activate_env gpu
nvidia-smi -L >&2 || die "no GPU visible on node '$(current_node)'" 6
# a GPU job that quietly runs on CPU is the failure mode to rule out first
python - <<'PY' || die "torch cannot see the GPU even though nvidia-smi can (is the CUDA module loaded?)" 6
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}, cuda available: {ok}" + (f", device: {torch.cuda.get_device_name(0)}" if ok else ""), file=sys.stderr)
sys.exit(0 if ok else 1)
PY
log "GPU smoke test"
python scripts/gpu_smoke.py
log "GPU-marked unit tests"
python -m pytest -p no:cacheprovider -m gpu "$@"
