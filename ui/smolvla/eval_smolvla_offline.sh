#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

POLICY_PATH="${POLICY_PATH:-${SMOLVLA_DEFAULT_OUTPUT_ROOT}/smolvla_task1_full/checkpoints/last/pretrained_model}"
REPO_ID="${REPO_ID:-${SMOLVLA_DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_DEFAULT_DATASET_ROOT}}"
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"

smolvla_repo_python examples/daksha/smolvla/eval_smolvla_offline.py \
  --policy-path "${POLICY_PATH}" \
  --dataset-repo-id "${REPO_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}"
