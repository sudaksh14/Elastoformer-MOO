#!/usr/bin/env bash
#SBATCH --job-name=em-build-fa
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only compile of flash-attn into the pylibs overlay. With ninja (parallel) the pessimistic runtime
# is 1 h, so submit with TIME_LIMIT=02:00:00 (2x), which scripts/submit.sh requires for build jobs:
#   TIME_LIMIT=02:00:00 scripts/submit.sh cpu submit_build_flash_attn.slurm
# No --mem: DAS-6 gives every job the whole node.
set -euo pipefail
echo "job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname -s)}"
exec "${ELASTOFORMER_ROOT:?ELASTOFORMER_ROOT not set; submit through scripts/submit.sh}/scripts/build_flash_attn.sh" "$@"
