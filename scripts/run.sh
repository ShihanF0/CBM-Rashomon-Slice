#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=TRUE

# Public experiment types: DivEns, random, ConvAda, Lora, Dropout.
EXP_TYPE="${EXP_TYPE:-Lora}"
DATA_NAME="${DATA_NAME:-HAM10000}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LAMBDA="${LAMBDA:-1}"
NUM_MODELS="${NUM_MODELS:-10}"
LR="${LR:-0.0001}"
WD="${WD:-0.0}"
SCHEDULER_STEP="${SCHEDULER_STEP:-10}"

ALPHA_MIN="${ALPHA_MIN:-0.5}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"

# LoRA masks are 12-bit transformer-block masks. ConvAda masks are 5-stage
# ResNet masks, e.g. 00000 or 1|11|11|00|00. Keep them separate.
LORA_MASK="${LORA_MASK:-000000000000}"
ADAPTER_MASK="${ADAPTER_MASK:-00000}"
if [[ -n "${MASK:-}" ]]; then
  LORA_MASK="$MASK"
  ADAPTER_MASK="$MASK"
fi

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
BOTTLENECK_DIM="${BOTTLENECK_DIM:-32}"
DROP_RATE="${DROP_RATE:-0.1}"
PASSTHROUGH="${PASSTHROUGH:-1}"

LOG_ROOT="${LOG_ROOT:-runs}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATA_NAME" in
  cifar10)
    DATA_DIR="annotated_cifar10_processed"
    DEFAULT_ENCODER="vit"
    N_ATTRIBUTES=143
    ;;
  CUB)
    DATA_DIR="CUB_200_2011"
    DEFAULT_ENCODER="vit"
    N_ATTRIBUTES=112
    ;;
  Awa2)
    DATA_DIR="Awa2"
    DEFAULT_ENCODER="vit"
    N_ATTRIBUTES=85
    ;;
  CelebA)
    DATA_DIR="CelebA"
    DEFAULT_ENCODER="vit"
    N_ATTRIBUTES=6
    ;;
  HAM10000)
    DATA_DIR="HAM10000"
    DEFAULT_ENCODER="vit"
    N_ATTRIBUTES=139
    ;;
  *)
    echo "error: unknown DATA_NAME=$DATA_NAME" >&2
    exit 2
    ;;
esac

if [[ -n "${ENCODER:-}" ]]; then
  SELECTED_ENCODER="$ENCODER"
else
  case "$EXP_TYPE" in
    Lora)
      SELECTED_ENCODER="vit"
      ;;
    ConvAda)
      SELECTED_ENCODER="resnet18"
      ;;
    DivEns|random|Dropout)
      SELECTED_ENCODER="$DEFAULT_ENCODER"
      ;;
    *)
      echo "error: unknown EXP_TYPE=$EXP_TYPE" >&2
      exit 2
      ;;
  esac
fi

case "$EXP_TYPE" in
  Lora)
    if [[ "$SELECTED_ENCODER" != "vit" && "$SELECTED_ENCODER" != "medical_vit" ]]; then
      echo "error: EXP_TYPE=Lora requires ENCODER=vit or ENCODER=medical_vit, got $SELECTED_ENCODER" >&2
      exit 2
    fi
    ACTIVE_MASK="$LORA_MASK"
    EXTRA_TAG="mask${ACTIVE_MASK}_r${LORA_R}_la${LORA_ALPHA}"
    ;;
  ConvAda)
    if [[ "$SELECTED_ENCODER" != "resnet18" ]]; then
      echo "error: EXP_TYPE=ConvAda requires ENCODER=resnet18, got $SELECTED_ENCODER" >&2
      exit 2
    fi
    ACTIVE_MASK="$ADAPTER_MASK"
    EXTRA_TAG="mask${ACTIVE_MASK}_bn${BOTTLENECK_DIM}"
    ;;
  Dropout)
    EXTRA_TAG="drop${DROP_RATE}_pass${PASSTHROUGH}"
    ;;
  DivEns|random)
    EXTRA_TAG="base"
    ;;
esac

RUN_NAME="seed${SEED}_enc${SELECTED_ENCODER}_m${NUM_MODELS}_lambda${LAMBDA}_alpha${ALPHA_MIN}-${ALPHA_MAX}_lr${LR}_wd${WD}_${EXTRA_TAG}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${DATA_NAME}/${EXP_TYPE}/${RUN_NAME}}"

TRAIN_ARGS=(
  -exp "$EXP_TYPE"
  -seed "$SEED"
  -log_dir "$LOG_DIR"
  -e "$EPOCHS"
  -pretrained
  -data_dir "$DATA_DIR"
  -n_attributes "$N_ATTRIBUTES"
  -batch_size "$BATCH_SIZE"
  -weight_decay "$WD"
  -lr "$LR"
  -scheduler_step "$SCHEDULER_STEP"
  -num_models "$NUM_MODELS"
  -lambda_c_acc "$LAMBDA"
  -dataname "$DATA_NAME"
  -encoder "$SELECTED_ENCODER"
  --alpha_min "$ALPHA_MIN"
  --alpha_max "$ALPHA_MAX"
  --early_stop_metric mean_child_val_miscls
)

case "$EXP_TYPE" in
  Lora)
    TRAIN_ARGS+=(
      -share_mask "$ACTIVE_MASK"
      --lora_r "$LORA_R"
      --lora_alpha "$LORA_ALPHA"
      --lora_dropout "$LORA_DROPOUT"
    )
    ;;
  ConvAda)
    TRAIN_ARGS+=(
      -share_mask "$ACTIVE_MASK"
      -bottleneck_dim "$BOTTLENECK_DIM"
    )
    ;;
  Dropout)
    TRAIN_ARGS+=(-dropout "$DROP_RATE")
    if [[ "$PASSTHROUGH" == "1" ]]; then
      TRAIN_ARGS+=(-passthrough)
    fi
    ;;
esac

cmd=(python -u train.py "${TRAIN_ARGS[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '+'
  printf ' %q' "${cmd[@]}"
  echo
else
  "${cmd[@]}"
fi
