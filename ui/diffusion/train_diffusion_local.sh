#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../smolvla/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

# ════════════════════════════════════════════════════════════════════════════════════
# Diffusion Policy Training Script for Daksha
# ════════════════════════════════════════════════════════════════════════════════════

REPO_ID="${REPO_ID:-${SMOLVLA_DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_DEFAULT_DATASET_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOLVLA_REPO_ROOT}/outputs/diffusion_daksha}"
JOB_NAME="${JOB_NAME:-diffusion_daksha}"
DEVICE="${DEVICE:-cuda}"
# torch.compile (inductor) needs a working Triton build; the generic PyPI triton wheel is ABI-
# mismatched against this Jetson/JetPack torch build (AttributeError: 'KernelMetadata' object
# has no attribute 'cluster_dims'), so compilation is off by default on this hardware.
COMPILE_MODEL="${COMPILE_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-50000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-20}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_REPO_ID="${HUB_REPO_ID:-}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-true}"
NUM_DIFFUSION_STEPS="${NUM_DIFFUSION_STEPS:-100}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
# Float32 frame timestamps can exceed the default 1e-4s tolerance on long, multi-episode
# concatenated video files; 0.001s still comfortably covers that quantization error.
TOLERANCE_S="${TOLERANCE_S:-0.001}"

echo "═══════════════════════════════════════════════════════════════════════════════════"
echo "Starting Diffusion Policy Training for Daksha"
echo "═══════════════════════════════════════════════════════════════════════════════════"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Job Name: ${JOB_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Steps: ${STEPS}"
echo "Diffusion Steps: ${NUM_DIFFUSION_STEPS}"
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

# Train Diffusion policy
split_args=()
if [[ -n "${EVAL_SPLIT:-}" ]]; then
  # eval_steps is set from EVAL_STEPS (the UI's eval_freq, independent of save_freq);
  # falls back to SAVE_FREQ only when run directly without the UI. With num_workers=0
  # and video-backed decoding, an *uncapped* eval pass over the whole held-out split
  # can take many minutes and looks like training hung -- max_eval_samples bounds it
  # to a small, fast, constant-size check (task-balanced) regardless of dataset size
  # or how low eval_steps is set.
  split_args+=(--dataset.eval_split="${EVAL_SPLIT}" --eval_steps="${EVAL_STEPS:-${SAVE_FREQ}}" \
    --max_eval_samples="${MAX_EVAL_SAMPLES:-64}")
fi
smolvla_repo_module lerobot.scripts.lerobot_train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=diffusion \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --policy.compile_model="${COMPILE_MODEL}" \
  --seed="${SEED}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  ${HUB_REPO_ID:+--policy.repo_id="${HUB_REPO_ID}"} \
  --policy.num_train_timesteps="${NUM_DIFFUSION_STEPS}" \
  --tolerance_s="${TOLERANCE_S}" "${split_args[@]}"

echo "✓ Diffusion training completed"
echo "Output saved to: ${OUTPUT_DIR}"
