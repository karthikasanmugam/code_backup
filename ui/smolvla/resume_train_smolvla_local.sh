#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-}"
RUN_PREFIX="${RUN_PREFIX:-smolvla_task1_full}"
SEARCH_ROOT="${SEARCH_ROOT:-${SMOLVLA_DEFAULT_OUTPUT_ROOT}}"
# Left empty by default so that, when not explicitly overridden, the original values
# saved in the checkpoint's train_config.json (loaded via --config_path) are kept as-is
# instead of being reset to arbitrary defaults.
NUM_WORKERS="${NUM_WORKERS:-}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
SEED="${SEED:-}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
LOG_FREQ="${LOG_FREQ:-}"
TOLERANCE_S="${TOLERANCE_S:-}"
# Set by the UI when resuming a checkpoint trained on another machine: the local run
# dir to keep saving into, and this machine's dataset copy (its saved root is stale).
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET_ROOT_OVERRIDE="${DATASET_ROOT_OVERRIDE:-}"

find_latest_config() {
  local run_prefix="$1"
  local search_root="$2"
  if [[ ! -d "${search_root}" ]]; then
    return 0
  fi
  find "${search_root}" -type f \
    -path "*/${run_prefix}*/checkpoints/*/pretrained_model/train_config.json" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="$(find_latest_config "${RUN_PREFIX}" "${SEARCH_ROOT}")"
  if [[ -z "${CONFIG_PATH}" && -d "${SEARCH_ROOT}" ]]; then
    CONFIG_PATH="$(find "${SEARCH_ROOT}" -type f -path "*/checkpoints/*/pretrained_model/train_config.json" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  fi
fi

if [[ -d "${CONFIG_PATH}" && -f "${CONFIG_PATH}/train_config.json" ]]; then
  CONFIG_PATH="${CONFIG_PATH}/train_config.json"
fi

if [[ -z "${CONFIG_PATH}" || ! -f "${CONFIG_PATH}" ]]; then
  echo "No smolvla train_config.json found for resume."
  echo "Pass CONFIG_PATH=/absolute/path/to/.../pretrained_model/train_config.json"
  echo "or set RUN_PREFIX to your run name."
  exit 1
fi

CONFIG_PATH="$(realpath "${CONFIG_PATH}")"
smolvla_maybe_ulimit

# --config_path restores every original training parameter from the checkpoint's
# train_config.json; the appends below only override the few the operator (or the UI,
# for per-machine runtime knobs) explicitly set. Anything left empty is inherited.
CMD=(
  lerobot.scripts.lerobot_train
  --config_path="${CONFIG_PATH}"
  --resume=true
)

if [[ -n "${SEED}" ]]; then
  CMD+=(--seed="${SEED}")
fi

if [[ -n "${NUM_WORKERS}" ]]; then
  CMD+=(--num_workers="${NUM_WORKERS}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
  CMD+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ -n "${STEPS}" ]]; then
  CMD+=(--steps="${STEPS}")
fi

if [[ -n "${SAVE_FREQ}" ]]; then
  CMD+=(--save_freq="${SAVE_FREQ}")
fi

if [[ -n "${LOG_FREQ}" ]]; then
  CMD+=(--log_freq="${LOG_FREQ}")
fi

if [[ -n "${TOLERANCE_S}" ]]; then
  CMD+=(--tolerance_s="${TOLERANCE_S}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir="${OUTPUT_DIR}")
fi

if [[ -n "${DATASET_ROOT_OVERRIDE}" ]]; then
  CMD+=(--dataset.root="${DATASET_ROOT_OVERRIDE}")
fi

smolvla_repo_module "${CMD[@]}"
