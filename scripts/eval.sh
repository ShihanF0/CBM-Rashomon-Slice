#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=TRUE

EXP_TYPE="${EXP_TYPE:-Lora}" # DivEns, random, ConvAda, Lora, or Dropout
DATA_NAME="${DATA_NAME:-cifar10}"  # CUB, cifar10, CelebA, or Awa2
TASK="${TASK:-similarity}" # SHAP, similarity, or task_acc
NUM_MODELS="${NUM_MODELS:-10}"
LOG_DIR="${LOG_DIR:-}"
if [[ -z "$LOG_DIR" ]]; then
  echo "error: LOG_DIR is required. Example: LOG_DIR=runs/cifar10/Lora/seed1_encvit_m10_lambda1_alpha0.5-1.0_lr0.0001_wd0.0_mask000000000000_r8_la16 bash scripts/eval.sh" >&2
  exit 2
fi

if [ "$DATA_NAME" == "cifar10" ]; then
  DATA_DIR="annotated_cifar10_processed"
  IMAGE_DIR=""
  DEFAULT_ENCODER="vit"
  N_ATTRIBUTES=143
elif [ "$DATA_NAME" == "CUB" ]; then
  DATA_DIR="CUB_200_2011"
  IMAGE_DIR=""
  DEFAULT_ENCODER="vit"
  N_ATTRIBUTES=112
elif [ "$DATA_NAME" == "Awa2" ]; then
  DATA_DIR="Awa2"
  IMAGE_DIR=""
  DEFAULT_ENCODER="vit"
  N_ATTRIBUTES=85
elif [ "$DATA_NAME" == "CelebA" ]; then
  DATA_DIR="CelebA"
  IMAGE_DIR=""
  DEFAULT_ENCODER="vit"
  N_ATTRIBUTES=6
else
  echo "error: unsupported DATA_NAME=$DATA_NAME" >&2
  exit 2
fi

if [[ -z "${ENCODER:-}" ]]; then
  case "$EXP_TYPE" in
    Lora)
      ENCODER="vit"
      ;;
    ConvAda)
      ENCODER="resnet18"
      ;;
    *)
      ENCODER="$DEFAULT_ENCODER"
      ;;
  esac
fi

LORA_MASK="${LORA_MASK:-000000000000}"
ADAPTER_MASK="${ADAPTER_MASK:-00000}"
if [[ -n "${MASK:-}" ]]; then
  LORA_MASK="$MASK"
  ADAPTER_MASK="$MASK"
fi
BOTTLENECK_DIM="${BOTTLENECK_DIM:-64}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
DROPOUT="${DROPOUT:-0.0625}"
PASSTHROUGH="${PASSTHROUGH:-1}"

if [[ "$EXP_TYPE" == "Lora" && "$ENCODER" != "vit" && "$ENCODER" != "medical_vit" ]]; then
  echo "error: EXP_TYPE=Lora requires ENCODER=vit or ENCODER=medical_vit" >&2
  exit 2
fi

if [[ "$EXP_TYPE" == "ConvAda" && "$ENCODER" != "resnet18" ]]; then
  echo "error: EXP_TYPE=ConvAda requires ENCODER=resnet18" >&2
  exit 2
fi

CMD=(
  python -u evaluation.py
  --task "$TASK"
  --exp "$EXP_TYPE"
  --log_dir "$LOG_DIR"
  --data_dir "$DATA_DIR"
  --image_dir "$IMAGE_DIR"
  --num_models "$NUM_MODELS"
  --dataname "$DATA_NAME"
  --encoder "$ENCODER"
)

case "$EXP_TYPE" in
  Lora)
    CMD+=(--share_mask "$LORA_MASK" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA")
    ;;
  ConvAda)
    CMD+=(--share_mask "$ADAPTER_MASK" --bottleneck_dim "$BOTTLENECK_DIM")
    ;;
  Dropout)
    CMD+=(--dropout "$DROPOUT")
    if [[ "$PASSTHROUGH" == "1" ]]; then
      CMD+=(--passthrough)
    fi
    ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
