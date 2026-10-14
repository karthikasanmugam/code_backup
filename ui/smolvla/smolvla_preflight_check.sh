#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_DEFAULT_DATASET_ROOT}}"
REPO_ID="${REPO_ID:-${SMOLVLA_DEFAULT_DATASET_REPO_ID}}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${SMOLVLA_DEFAULT_PRETRAINED_PATH}}"

echo "=== smolvla Preflight ==="
echo "repo_root: ${SMOLVLA_REPO_ROOT}"
echo "dataset_root: ${DATASET_ROOT}"
echo "dataset_repo_id: ${REPO_ID}"
echo "pretrained_path: ${PRETRAINED_PATH}"
echo ""

smolvla_repo_python examples/daksha/smolvla/check_dataset_smolvla.py \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${REPO_ID}"

echo ""
echo "=== Trainer Import Check ==="
smolvla_repo_python examples/daksha/smolvla/run_lerobot_train_smolvla.py --help >/dev/null
echo "smolvla train wrapper imports correctly."

echo ""
echo "=== Runtime Checks ==="
# The torch/lerobot stack now runs under the global Python (no per-repo venv).
echo "interpreter: $(smolvla_repo_python -c 'import sys; print(sys.executable)')"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1
else
  echo "ffmpeg: not found"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "nvidia-smi: not found"
fi

echo ""
echo "Preflight passed."
