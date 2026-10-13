#!/usr/bin/env bash
# Read-only inspection of the DAS-6 environment. Run it ON DAS-6 and keep the output:
#
#   bash scripts/inspect_das6.sh 2>&1 | tee inspect_das6.txt
#
# It changes nothing. Section 6 submits a short job only if you set RUN_SRUN=1.
set -uo pipefail

section() { printf '\n===== %s =====\n' "$*"; }

section "1. identity"
date; hostname; whoami; echo "HOME=$HOME"

section "2. storage"
ls -ld /home/skalra /var/scratch/skalra 2>&1
df -h /home/skalra /var/scratch/skalra 2>&1
quota -s 2>&1 | head -5
echo "writable scratch: $( [ -w /var/scratch/skalra ] && echo yes || echo no )"

section "3. python environments (conda and venv)"
command -v conda && conda env list 2>&1
command -v python3 && python3 --version
echo "-- venv candidates (pyvenv.cfg within 4 levels of \$HOME and /var/scratch/skalra)"
find "$HOME" /var/scratch/skalra -maxdepth 4 -name pyvenv.cfg 2>/dev/null | sed 's|/pyvenv.cfg||' | head -30
echo "-- directories named prune_llm or matryoshka_mm"
find "$HOME" /var/scratch/skalra -maxdepth 4 -type d \( -name prune_llm -o -name matryoshka_mm \) 2>/dev/null

probe() {  # probe <python executable>
  "$1" - <<'PY' 2>&1
import importlib, sys
print("python", sys.version.split()[0])
for m in ["torch","transformers","tokenizers","safetensors","numpy","scipy","yaml","pytest","datasets","accelerate","peft","lm_eval","bitsandbytes"]:
    try:
        mod = importlib.import_module(m); print(" ", m, getattr(mod, "__version__", "?"))
    except Exception:
        print(" ", m, "MISSING")
try:
    import torch; print("  cuda_available", torch.cuda.is_available(), "torch_cuda", torch.version.cuda)
except Exception:
    pass
PY
}
for name in prune_llm matryoshka_mm; do
  echo "-- probing $name"
  py=""
  for base in "$HOME" "$HOME/venvs" "$HOME/.virtualenvs" /var/scratch/skalra; do
    [ -x "$base/$name/bin/python" ] && py="$base/$name/bin/python" && break
  done
  if [ -z "$py" ] && command -v conda >/dev/null 2>&1; then
    prefix="$(conda env list 2>/dev/null | awk -v n="$name" '$1==n {print $NF}')"
    [ -n "$prefix" ] && py="$prefix/bin/python"
  fi
  if [ -n "$py" ]; then echo "found: $py"; probe "$py"; else echo "NOT FOUND as venv or conda env"; fi
done

section "4. slurm configuration"
command -v sbatch sinfo squeue 2>&1
sinfo -o '%P %a %l %D %N %G %f' 2>&1
echo "-- per node (name partition gres cpus mem state)"
sinfo -N -o '%N %P %G %c %m %t' 2>&1 | head -80
scontrol show config 2>&1 | grep -E 'ClusterName|SelectType|AccountingStorageType|DefaultTime|MaxTime'
sacctmgr -n show assoc user=skalra format=account,partition,qos 2>&1 | head -10

section "5. the nodes the policy cares about"
for n in 201 202 203 204 205 206 207 208; do
  scontrol show node "node$n" 2>&1 | grep -E 'NodeName|Gres=|CPUTot|RealMemory|State=' | tr '\n' ' '; echo
done

section "6. optional GPU placement check (RUN_SRUN=1 to enable)"
if [ "${RUN_SRUN:-0}" = 1 ]; then
  srun --nodelist=node205 --gres=gpu:1 --time=00:02:00 bash -c 'hostname; nvidia-smi -L'
else
  echo "skipped; rerun with RUN_SRUN=1 to submit a 2-minute job on node205"
fi

section "7. network and model access"
for url in https://huggingface.co https://pypi.org; do
  printf '%s -> ' "$url"; curl -s -o /dev/null -m 10 -w '%{http_code}\n' "$url" 2>&1 || echo "unreachable"
done
echo "HF_TOKEN set: $( [ -n "${HF_TOKEN:-}" ] && echo yes || echo no )"
echo
echo "done"
