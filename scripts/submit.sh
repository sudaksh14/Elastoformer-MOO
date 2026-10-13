#!/usr/bin/env bash
# Submit a job restricted to the allowed DAS-6 nodes.
#
#   scripts/submit.sh <gpu|cpu> <slurm-script> [script args...]
#
# GPU jobs may only run on nodes 205-208, CPU-only jobs on nodes 201-204. Slurm's --nodelist
# forces every listed node into the job, so the restriction is expressed as --exclude of the
# complement, computed from the cluster's real node list. The job scripts also check the node
# they land on and exit before doing any work if it is not allowed.
#
# Environment:
#   TIME_LIMIT     e.g. 01:00:00 (defaults: cpu tests 00:30:00, gpu tests 00:20:00; required for experiments)
#   GPU_GRES       gres string for GPU jobs (default gpu:1; `gpu:A40:1` selects an A40, see `sinfo -o %G`)
#   PARTITION      sbatch partition (defaults: gpu -> defq, cpu -> fatq)
#   ALL_NODES      space-separated node names; overrides `sinfo` (used for dry runs and tests)
#   DRY_RUN=1      print the sbatch command and exit without submitting
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"


[[ $# -ge 2 ]] || die "usage: $0 <gpu|cpu> <slurm-script> [script args...]" 2
kind="$1"; script="$2"; shift 2
[[ "$kind" == gpu || "$kind" == cpu ]] || die "kind must be gpu or cpu, got '$kind'" 2
[[ -f "$ELASTOFORMER_ROOT/$script" || -f "$script" ]] || die "slurm script not found: $script" 2

if [[ -n "${ALL_NODES:-}" ]]; then
  nodes="$ALL_NODES"
else
  command -v sinfo >/dev/null 2>&1 || die "sinfo not found; run this on the DAS-6 head node (or set ALL_NODES for a dry run)" 5
  nodes="$(sinfo -N -h -o '%N' | sort -u | tr '\n' ' ')"
fi
exclude="$(python3 "$ELASTOFORMER_ROOT/scripts/node_policy.py" exclude "$kind" "$nodes")" \
  || die "cannot build a safe node restriction for $kind jobs; not submitting" 3

# Time limits are 2x a pessimistic runtime estimate and must never exceed 150 h. Test scripts have
# defaults (cpu: pessimistic 15 min -> 30 min; gpu: pessimistic 10 min -> 20 min); experiments do not,
# because their runtime depends on the model and step count, so TIME_LIMIT is required for them.
case "$script" in
  *experiments*|*build*) [[ -n "${TIME_LIMIT:-}" ]] || die "set TIME_LIMIT (2x your pessimistic runtime estimate) for experiment and build jobs" 2 ;;
esac
# Admin rule: never request memory on DAS-6; every job occupies the whole node. sbatch also reads these
# from the environment, so make sure none leaks in.
unset SBATCH_MEM_PER_NODE SBATCH_MEM_PER_CPU SBATCH_MEM_PER_GPU SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
default_time=00:30:00; [[ "$kind" == gpu ]] && default_time=00:20:00
# DAS-6: defq = GPU nodes, fatq = CPU nodes. GPU work always queues for defq; it never falls back to CPU nodes.
default_partition=fatq; [[ "$kind" == gpu ]] && default_partition=defq
# No --mem, ever: by the cluster admins' rule every job occupies the whole node (and the nodes report RealMemory=1).
args=(--exclude="$exclude" --time="${TIME_LIMIT:-$default_time}" --partition="${PARTITION:-$default_partition}"
      --output="$SCRATCH_ROOT/logs/%x_%j.out" --export="ALL,ELASTOFORMER_ROOT=$ELASTOFORMER_ROOT,SCRATCH_ROOT=$SCRATCH_ROOT")
[[ "$kind" == gpu ]] && args+=(--gres="${GPU_GRES:-gpu:1}")

cmd=(sbatch "${args[@]}" "$script" "$@")
log "kind=$kind  excluded ${exclude//[^,]/} (comma count)  command:"
printf '  %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then log "DRY_RUN=1: not submitting"; exit 0; fi
ensure_scratch
cd "$ELASTOFORMER_ROOT"
exec "${cmd[@]}"
