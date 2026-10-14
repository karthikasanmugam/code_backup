#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-}"
RUN_PREFIX="${RUN_PREFIX:-smolvla_task1_full}"
SEARCH_ROOT="${SEARCH_ROOT:-${SMOLVLA_DEFAULT_OUTPUT_ROOT}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SEED="${SEED:-1000}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
LOG_FREQ="${LOG_FREQ:-}"
TOLERANCE_S="${TOLERANCE_S:-}"

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

CMD=(
  examples/daksha/smolvla/run_lerobot_train_smolvla.py
  --config_path="${CONFIG_PATH}"
  --resume=true
  --seed="${SEED}"
  --num_workers="${NUM_WORKERS}"
  --dataset.video_backend="${VIDEO_BACKEND}"
)

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

smolvla_repo_python "${CMD[@]}"
