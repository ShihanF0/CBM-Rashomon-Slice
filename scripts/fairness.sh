#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

EXP_TYPE="${EXP_TYPE:-Lora}"
LOG_DIR="${LOG_DIR:-}"
DATA_DIR="${DATA_DIR:-}"

ENCODER="${ENCODER:-vit}"
N_ATTRIBUTES="${N_ATTRIBUTES:-6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_MODELS="${NUM_MODELS:-10}"
EXPAND_DIM="${EXPAND_DIM:-0}"

LORA_MASK="${LORA_MASK:-000000000000}"
ADAPTER_MASK="${ADAPTER_MASK:-00000}"
if [[ -n "${MASK:-}" ]]; then
  LORA_MASK="$MASK"
  ADAPTER_MASK="$MASK"
fi
BOTTLENECK_DIM="${BOTTLENECK_DIM:-32}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

PROTECTED_ATTR="${PROTECTED_ATTR:-Young}"
TARGET_ATTR="${TARGET_ATTR:-Wavy_Hair}"
OUT_DIR="${OUT_DIR:-}"

if [[ -z "$LOG_DIR" || -z "$DATA_DIR" ]]; then
  echo "error: set LOG_DIR and DATA_DIR before running fairness.sh" >&2
  echo "example: LOG_DIR=runs/CelebA/Lora/... DATA_DIR=CelebA DRY_RUN=1 bash scripts/fairness.sh" >&2
  exit 2
fi

if [[ "$EXP_TYPE" == "Lora" && "$ENCODER" != "vit" && "$ENCODER" != "medical_vit" ]]; then
  echo "error: EXP_TYPE=Lora requires ENCODER=vit or ENCODER=medical_vit" >&2
  exit 2
fi

if [[ "$EXP_TYPE" == "ConvAda" && "$ENCODER" != "resnet18" ]]; then
  echo "error: EXP_TYPE=ConvAda requires ENCODER=resnet18" >&2
  exit 2
fi

CMD=(
  python -u experiments/fairness.py
  --exp "$EXP_TYPE"
  --log_dir "$LOG_DIR"
  --data_dir "$DATA_DIR"
  --num_models "$NUM_MODELS"
  --encoder "$ENCODER"
  --n_attributes "$N_ATTRIBUTES"
  --expand_dim "$EXPAND_DIM"
  --batch_size "$BATCH_SIZE"
  --protected_attr "$PROTECTED_ATTR"
  --target_attr "$TARGET_ATTR"
)

if [[ -n "$OUT_DIR" ]]; then
  CMD+=(--out_dir "$OUT_DIR")
fi

case "$EXP_TYPE" in
  Lora)
    CMD+=(--share_mask "$LORA_MASK" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
    ;;
  ConvAda)
    CMD+=(--share_mask "$ADAPTER_MASK" --bottleneck_dim "$BOTTLENECK_DIM")
    ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
