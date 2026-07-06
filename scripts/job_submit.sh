#!/usr/bin/env bash
set -euo pipefail

# Sweep grid for LR and ALPHA_MIN by repeatedly submitting run.sh via sbatch.
# run.sh must read LR/ALPHA_MIN from the environment (it does via ${VAR:-default}).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_SH="${RUN_SH:-$SCRIPT_DIR/run.sh}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

# Edit these grids as needed.
LRS=(0.0001 0.0003)
ALPHA_MINS=(0.25 0.5)

# Optional extra sbatch flags, e.g.:
SBATCH_ARGS=" --gres=gpu:1"
SBATCH_ARGS_STR="${SBATCH_ARGS:-}"
SBATCH_ARGS_ARR=()
if [[ -n "$SBATCH_ARGS_STR" ]]; then
  # shellcheck disable=SC2206
  SBATCH_ARGS_ARR=($SBATCH_ARGS_STR)
fi

mkdir -p "$REPO_ROOT/logs"

if [[ ! -f "$RUN_SH" ]]; then
  echo "error: cannot find run.sh at: $RUN_SH" >&2
  echo "tip: set RUN_SH to an absolute path if you want to submit a different training wrapper." >&2
  exit 2
fi

for lr in "${LRS[@]}"; do
  for alpha_min in "${ALPHA_MINS[@]}"; do
    lr_tag="${lr//./p}"
    alpha_tag="${alpha_min//./p}"
    job_name="CBM_lr${lr_tag}_amin${alpha_tag}"

    cmd=(
      sbatch
      "${SBATCH_ARGS_ARR[@]}"
      --job-name="$job_name"
      --chdir="$REPO_ROOT"
      --export="ALL,LR=$lr,ALPHA_MIN=$alpha_min"
      "$RUN_SH"
    )

    if (( DRY_RUN )); then
      printf '+ %q ' "${cmd[@]}"
      echo
    else
      "${cmd[@]}"
    fi
  done
done
