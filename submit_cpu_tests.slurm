#!/usr/bin/env bash
#SBATCH --job-name=em-cpu-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Submit with:  scripts/submit.sh cpu submit_cpu_tests.slurm
# (the wrapper adds --exclude so only nodes 201-204 are eligible; run_cpu_tests.sh re-checks)
set -euo pipefail
echo "job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname -s)}"
exec "${ELASTOFORMER_ROOT:?ELASTOFORMER_ROOT not set; submit through scripts/submit.sh}/run_cpu_tests.sh" "$@"
