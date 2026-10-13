#!/usr/bin/env bash
#SBATCH --job-name=em-gpu-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Submit with:  scripts/submit.sh gpu submit_gpu_tests.slurm
# (the wrapper adds --exclude so only nodes 205-208 are eligible and requests the GPU;
#  run_gpu_tests.sh re-checks the node before touching the GPU)
set -euo pipefail
echo "job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname -s)}"
exec "${ELASTOFORMER_ROOT:?ELASTOFORMER_ROOT not set; submit through scripts/submit.sh}/run_gpu_tests.sh" "$@"
