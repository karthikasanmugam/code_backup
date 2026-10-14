#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../smolvla/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

# ════════════════════════════════════════════════════════════════════════════════════
# Pi-0.5 Training Script for Daksha
# ════════════════════════════════════════════════════════════════════════════════════

REPO_ID="${REPO_ID:-${SMOLVLA_DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_DEFAULT_DATASET_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOLVLA_REPO_ROOT}/outputs/pi_0_5_daksha}"
JOB_NAME="${JOB_NAME:-pi_0_5_daksha}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
# torch.compile (inductor) needs a working Triton build; the generic PyPI triton wheel is ABI-
# mismatched against this Jetson/JetPack torch build (AttributeError: 'KernelMetadata' object
# has no attribute 'cluster_dims'), so compilation is off by default on this hardware.
COMPILE_MODEL="${COMPILE_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-20}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-true}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
# Float32 frame timestamps can exceed the default 1e-4s tolerance on long, multi-episode
# concatenated video files; 0.001s still comfortably covers that quantization error.
TOLERANCE_S="${TOLERANCE_S:-0.001}"

echo "═══════════════════════════════════════════════════════════════════════════════════"
echo "Starting Pi-0.5 Training for Daksha"
echo "═══════════════════════════════════════════════════════════════════════════════════"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Job Name: ${JOB_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Steps: ${STEPS}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "═══════════════════════════════════════════════════════════════════════════════════"

# Dataset validation
if [[ "${RUN_DATASET_CHECK}" == "true" ]]; then
  echo "🔍 Running dataset validation..."
  smolvla_repo_python examples/daksha/smolvla/check_dataset_smolvla.py \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-repo-id "${REPO_ID}" || {
    echo "❌ Dataset validation failed"
    exit 1
  }
  echo "✓ Dataset validation passed"
fi

# Create output directory
if [[ -d "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  echo "⚠️  Output directory exists, using: ${OUTPUT_DIR}"
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

smolvla_maybe_ulimit

echo "🚀 Starting training process..."

# Train Pi-0.5 policy
smolvla_repo_module lerobot.scripts.lerobot_train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=pi05 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --policy.dtype="${DTYPE}" \
  --policy.compile_model="${COMPILE_MODEL}" \
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  --seed="${SEED}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.hidden_dim="${HIDDEN_DIM}" \
  --tolerance_s="${TOLERANCE_S}"

echo "✓ Pi-0.5 training completed"
echo "Output saved to: ${OUTPUT_DIR}"
