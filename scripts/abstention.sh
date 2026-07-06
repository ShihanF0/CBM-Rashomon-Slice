#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

DATA_NAME="${DATA_NAME:-HAM10000}"
case "$DATA_NAME" in
  HAM10000)
    DEFAULT_RUNS_JSON="experiments/abstention_HAM10000.json"
    DEFAULT_DATA_DIR="HAM10000"
    DEFAULT_ENCODER="vit"
    DEFAULT_BATCH_SIZE="32"
    DEFAULT_LORA_R="16"
    DEFAULT_LORA_ALPHA="32"
    ;;
  Awa2)
    DEFAULT_RUNS_JSON="experiments/abstention_Awa2.json"
    DEFAULT_DATA_DIR="Awa2"
    DEFAULT_ENCODER="vit"
    DEFAULT_BATCH_SIZE="32"
    DEFAULT_LORA_R="8"
    DEFAULT_LORA_ALPHA="16"
    ;;
  *)
    echo "error: unsupported DATA_NAME for abstention.sh: $DATA_NAME" >&2
    echo "supported: HAM10000, Awa2" >&2
    exit 2
    ;;
esac

RUNS_JSON="${RUNS_JSON:-$DEFAULT_RUNS_JSON}"
if [[ ! -f "$RUNS_JSON" ]]; then
  echo "error: RUNS_JSON not found: $RUNS_JSON" >&2
  echo "tip: set RUNS_JSON=/path/to/runs.json, or use experiments/abstention_HAM10000.json." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
ENCODER="${ENCODER:-$DEFAULT_ENCODER}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
LORA_R="${LORA_R:-$DEFAULT_LORA_R}"
LORA_ALPHA="${LORA_ALPHA:-$DEFAULT_LORA_ALPHA}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
RASH_LOG="${OUT_DIR:-abstention_results_${DATA_NAME}}"

if [[ "$ENCODER" != "vit" && "$ENCODER" != "medical_vit" && "$ENCODER" != "resnet18" ]]; then
  echo "error: unsupported ENCODER=$ENCODER" >&2
  exit 2
fi

COMMON=(
  --runs_json "$RUNS_JSON"
  --dataname "$DATA_NAME"
  --data_dir "$DATA_DIR"
  --batch_size "$BATCH_SIZE"
  --encoder "$ENCODER"
  --sg_alpha 5.0
  --sg_score entropy
  --orig_mode mc
  --orig_mc_samples 64
  --class_idxs=-1
  --eval_splits val,test
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --out_dir "$RASH_LOG"
)

TAUS="${TAUS:-0.80,0.825,0.85,0.875,0.90,0.925,0.95,0.975}"
BUDGETS="${BUDGETS:-5}"
CLASS_SWEEP="${CLASS_SWEEP:-3}"
# TAUS="0.90"
# BUDGETS="0,1,2,3,4,5,6,7"

CMD=(
  python -u experiments/abstention.py
  "${COMMON[@]}"
  --sweep_taus "${TAUS}" \
  --sweep_budgets "${BUDGETS}" \
  --sweep_class_idxs "${CLASS_SWEEP}" \
  --overlap uncertain \
  --uncertain_policy intersection
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
