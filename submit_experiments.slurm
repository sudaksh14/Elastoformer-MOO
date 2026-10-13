#!/usr/bin/env bash
#SBATCH --job-name=em-experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# One experiment stage per job. Submit with, for example:
#   MODEL=llama3.2-1b TIME_LIMIT=12:00:00 scripts/submit.sh gpu submit_experiments.slurm train
# Stages: permute | train | eval  (all GPU work, so always submit with `gpu`).
# Run one stage, check its log, then submit the next; do not submit the whole pipeline at once.
set -euo pipefail
echo "job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname -s)} stage: ${1:-<none>}"
exec "${ELASTOFORMER_ROOT:?ELASTOFORMER_ROOT not set; submit through scripts/submit.sh}/run_experiments.sh" "$@"
