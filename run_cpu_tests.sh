#!/usr/bin/env bash
# CPU-only tests for DAS-6 nodes 201-203. Exits 3 on any other node before running anything.
#
#   ./run_cpu_tests.sh [pytest args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/lib.sh

require_node_class cpu
ensure_scratch
activate_env
log "running CPU tests (GPU-marked tests are excluded)"
CUDA_VISIBLE_DEVICES="" python -m pytest -p no:cacheprovider -m "not gpu" "$@"
