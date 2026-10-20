#!/usr/bin/env bash
# One stage of the experiment pipeline for one model. Heavy artifacts go to $SCRATCH_ROOT.
#
#   MODEL=smollm2-360m ./run_experiments.sh download   # CPU node: fetch weights into the HF cache (HF_TOKEN if gated)
#   MODEL=smollm2-360m ./run_experiments.sh data       # CPU node: WikiText-2 / C4 token files for this tokenizer
#   MODEL=smollm2-360m ./run_experiments.sh train-data # CPU node: tokenize the training corpus (FineWeb-Edu from smollm-corpus)
#   MODEL=smollm2-360m ./run_experiments.sh e1e2       # GPU node: E1 + E2 (ATTN=sdpa to avoid flash attention)
#   MODEL=smollm2-360m ./run_experiments.sh bench      # GPU node: sdpa vs flash_attention_2 timings
#   NPROC=2 ./run_experiments.sh distsmoke             # GPU node with 2 GPUs: multi-GPU smoke test (GPU_GRES=gpu:A40:2)
#   NPROC=2 MODEL=... ./run_experiments.sh train       # data-parallel training, one process per GPU
#   MODEL=llama3.2-1b ./run_experiments.sh check-models
#   MODEL=llama3.2-1b ./run_experiments.sh permute     # needs $SCRATCH_ROOT/data/calib.bin
#   MODEL=llama3.2-1b ./run_experiments.sh train       # needs $SCRATCH_ROOT/data/{train,val}.bin
#   MODEL=llama3.2-1b ./run_experiments.sh eval
#
# Stages e1e2, bench, permute, train and eval are GPU work: they exit 3 on any node outside 205-208.
# Stages download and data are CPU work: they exit 3 on any node outside 201-203.
# Flash attention is explicit (ATTN, default flash_attention_2) and never silently replaced by sdpa.
# Environment:
#   MODEL     registry name (default llama3.2-1b)
#   CONFIG    training config (default configs/experiments/<MODEL>_nested.yaml, else smoke.yaml)
#   RUN_ID    run directory name under $SCRATCH_ROOT/runs (default <MODEL>_<timestamp>);
#             an existing run directory is never overwritten unless RESUME=1
#   RESUME=1  resume the run in RUN_ID
# Extra arguments after the stage are passed through as --set key=value overrides (train only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/lib.sh

[[ $# -ge 1 ]] || die "usage: [MODEL=...] $0 <check-models|permute|train|eval> [key=value ...]" 2
stage="$1"; shift
MODEL="${MODEL:-llama3.2-1b}"
PERMUTED="$SCRATCH_ROOT/models/${MODEL}-permuted"

case "$stage" in
  check-models)
    activate_env
    ensure_scratch
    python scripts/check_models.py --out "$SCRATCH_ROOT/runs/model_facts_$(date +%Y%m%d).json"
    ;;
  download)
    require_node_class cpu; ensure_scratch; activate_env
    python scripts/download_model.py --model "$MODEL"
    ;;
  data)
    require_node_class cpu; ensure_scratch; activate_env
    python scripts/prepare_eval_data.py --tokenizer "$MODEL" --out-dir "$SCRATCH_ROOT/data/$MODEL"
    ;;
  train-data)
    require_node_class cpu; ensure_scratch; activate_env
    python scripts/prepare_train_data.py --tokenizer "$MODEL" --out-dir "$SCRATCH_ROOT/data/$MODEL" "$@"
    ;;
  e1e2)
    require_node_class gpu; ensure_scratch; activate_env gpu
    RUN_ID="${RUN_ID:-e1e2_${MODEL}_$(date +%Y%m%d_%H%M%S)}"
    RUN_DIR="$SCRATCH_ROOT/runs/$RUN_ID"
    [[ ! -e "$RUN_DIR" ]] || die "$RUN_DIR already exists; choose a new RUN_ID" 5
    log "E1/E2 for $MODEL, attention=${ATTN:-flash_attention_2}, run=$RUN_DIR"
    python scripts/run_e1_e2.py --model "$MODEL" --data-dir "$SCRATCH_ROOT/data/$MODEL" --out "$RUN_DIR" \
      --attn "${ATTN:-flash_attention_2}" --device cuda "$@"
    # the training stage reads the importance-ordered checkpoint from $PERMUTED: publish it once, never overwrite
    if [[ ! -d "$PERMUTED" && -d "$RUN_DIR/permuted" ]]; then
      mkdir -p "$(dirname "$PERMUTED")"
      cp -r "$RUN_DIR/permuted" "$PERMUTED"
      log "published the importance-ordered checkpoint to $PERMUTED"
    fi
    ;;
  bench)
    require_node_class gpu; ensure_scratch; activate_env gpu
    tokens_args=()
    [[ -f "$SCRATCH_ROOT/data/$MODEL/wikitext2_test.bin" ]] && tokens_args=(--tokens "$SCRATCH_ROOT/data/$MODEL/wikitext2_test.bin")
    python scripts/bench_attention.py --model "$MODEL" "${tokens_args[@]}" \
      --out "$SCRATCH_ROOT/runs/bench_attention_${MODEL}_$(date +%Y%m%d_%H%M%S).json" "$@"
    ;;
  permute)
    require_node_class gpu; ensure_scratch; activate_env gpu
    CALIB="$SCRATCH_ROOT/data/$MODEL/calib.bin"; [[ -f "$CALIB" ]] || CALIB="$SCRATCH_ROOT/data/calib.bin"
    [[ -f "$CALIB" ]] || die "missing $SCRATCH_ROOT/data/$MODEL/calib.bin (run: MODEL=$MODEL ./run_experiments.sh data)" 5
    [[ ! -e "$PERMUTED" ]] || die "$PERMUTED already exists; remove it or choose another MODEL" 5
    python scripts/permute_model.py --model "$MODEL" --calib "$CALIB" --out "$PERMUTED" --device cuda "$@"
    ;;
  train)
    require_node_class gpu; ensure_scratch; activate_env gpu
    CONFIG="${CONFIG:-configs/experiments/${MODEL}_nested.yaml}"
    [[ -f "$CONFIG" ]] || CONFIG="configs/experiments/smoke.yaml"
    RUN_ID="${RUN_ID:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
    RUN_DIR="$SCRATCH_ROOT/runs/$RUN_ID"
    if [[ -e "$RUN_DIR" && "${RESUME:-0}" != 1 ]]; then
      die "$RUN_DIR already exists; set RESUME=1 to continue it or choose a new RUN_ID" 5
    fi
    [[ -d "$PERMUTED" ]] || log "warning: $PERMUTED not found; training from the unpermuted registry model"
    D="$SCRATCH_ROOT/data/$MODEL"   # per-tokenizer token files from prepare_train_data.py / prepare_eval_data.py
    [[ -f "$D/train.bin" ]] || die "missing $D/train.bin (run: MODEL=$MODEL ./run_experiments.sh train-data)" 5
    # Read the data from the node-local NVMe instead of NFS: copied once at job start, removed at exit.
    # STAGE_LOCAL=0 disables it; STAGE_ROOT overrides the location (default /local/$USER).
    if [[ "${STAGE_LOCAL:-1}" == 1 ]]; then
      if stage_data "$D" "${STAGE_ROOT:-/local/${USER:-$(id -un)}}/em_${RUN_ID}" train.bin wikitext2_test.bin c4_val.bin fineweb_val.bin; then
        D="$STAGED_DIR"
      else
        log "warning: could not stage to node-local storage; reading the shared copy"
      fi
    fi
    sets=(--set "out_dir=$RUN_DIR" --set "data.train_tokens=$D/train.bin"
          --set "data.val_tokens.wikitext2=$D/wikitext2_test.bin" --set "data.val_tokens.c4=$D/c4_val.bin"
          --set "data.val_tokens.fineweb=$D/fineweb_val.bin"
          --set "resume=$([[ "${RESUME:-0}" == 1 ]] && echo true || echo false)")
    [[ -d "$PERMUTED" ]] && sets+=(--set "model=$PERMUTED") || sets+=(--set "model=$MODEL")
    for kv in "$@"; do sets+=(--set "$kv"); done
    # NPROC > 1: one process per GPU under torchrun (the job must own that many GPUs, e.g. GPU_GRES=gpu:A40:2)
    launch=(python)
    if [[ "${NPROC:-1}" -gt 1 ]]; then
      export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      launch=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
    fi
    log "config=$CONFIG run=$RUN_DIR processes=${NPROC:-1}"
    "${launch[@]}" scripts/train_nested.py --config "$CONFIG" "${sets[@]}"
    ;;
  distsmoke)
    require_node_class gpu; ensure_scratch; activate_env gpu
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    python -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-2}" scripts/dist_smoke.py
    ;;
  rescore)
    # Re-score a checkpoint on every block of the evaluation files (one protocol for all runs). MODEL_DIR is a run's
    # final/ directory or the permuted model; RUN_DIR (optional) gives the run's train_config.json (HV exclusion).
    require_node_class gpu; ensure_scratch; activate_env gpu
    [[ -n "${MODEL_DIR:-}" ]] || die "set MODEL_DIR to a checkpoint directory" 2
    OUT="${OUT:-${RUN_DIR:-$SCRATCH_ROOT/runs/rescore}/eval_full.json}"
    rescore_args=(--model "$MODEL_DIR" --data-dir "$SCRATCH_ROOT/data/$MODEL" --out "$OUT")
    [[ -n "${RUN_DIR:-}" ]] && rescore_args+=(--run-dir "$RUN_DIR")
    python scripts/rescore.py "${rescore_args[@]}" "$@"
    ;;
  eval)
    require_node_class gpu; ensure_scratch; activate_env gpu
    # TOKENS: space-separated token files (default $SCRATCH_ROOT/data/val.bin); extra arguments go to eval_ppl.py.
    # CKPT overrides the model (default: the importance-ordered checkpoint if present, else the registry model).
    model_dir="${CKPT:-$PERMUTED}"; [[ -d "$model_dir" || -n "${CKPT:-}" ]] || model_dir="$MODEL"
    for tokens in ${TOKENS:-$SCRATCH_ROOT/data/val.bin}; do
      tag="$(basename "$tokens" .bin)"
      python scripts/eval_ppl.py --model "$model_dir" --tokens "$tokens" --device cuda \
        --out "$SCRATCH_ROOT/runs/eval_${MODEL}_${tag}_$(date +%Y%m%d_%H%M%S).json" "$@"
    done
    ;;
  *) die "unknown stage '$stage' (download|data|train-data|e1e2|bench|distsmoke|check-models|permute|train|rescore|eval)" 2 ;;
esac
