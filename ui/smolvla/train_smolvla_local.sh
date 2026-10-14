#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

REPO_ID="${REPO_ID:-${SMOLVLA_DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-${SMOLVLA_DEFAULT_DATASET_ROOT}}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${SMOLVLA_DEFAULT_PRETRAINED_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOLVLA_REPO_ROOT}/outputs/smolvla_daksha}"
JOB_NAME="${JOB_NAME:-smolvla_daksha}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
# torch.compile (inductor) needs a working Triton build; the generic PyPI triton wheel is ABI-
# mismatched against this Jetson/JetPack torch build (AttributeError: 'KernelMetadata' object
# has no attribute 'cluster_dims'), so compilation is off by default on this hardware.
COMPILE_MODEL="${COMPILE_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# Future actions predicted per inference (chunk_size); n_action_steps must not exceed it.
ACTION_HORIZON="${ACTION_HORIZON:-20}"
STEPS="${STEPS:-50000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-20}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RUN_DATASET_CHECK="${RUN_DATASET_CHECK:-true}"
# Float32 frame timestamps can exceed the default 1e-4s tolerance on long, multi-episode
# concatenated video files; 0.001s still comfortably covers that quantization error.
TOLERANCE_S="${TOLERANCE_S:-0.001}"

if [[ "${RUN_DATASET_CHECK}" == "true" ]]; then
  check_args=()
  if [[ "${REPAIR_MISSING_QUANTILES:-false}" == "true" ]]; then
    check_args+=(--repair-missing-quantiles)
  fi
  smolvla_repo_python examples/daksha/smolvla/check_dataset_smolvla.py \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-repo-id "${REPO_ID}" "${check_args[@]}"
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  echo "Output directory exists, switching to: ${OUTPUT_DIR}"
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
smolvla_maybe_ulimit

split_args=()
if [[ -n "${EVAL_SPLIT:-}" ]]; then
  # eval_steps is set from EVAL_STEPS, which app.py derives from the UI's eval_freq
  # (independent of save_freq). With num_workers=0 and video-backed decoding, an
  # *uncapped* eval pass over the whole held-out split can take many minutes and looks
  # like training hung -- max_eval_samples bounds it to a small, fast, constant-size
  # check (task-balanced) regardless of dataset size or eval_freq.
  split_args+=(--dataset.eval_split="${EVAL_SPLIT}" --eval_steps="${EVAL_STEPS:-500}" \
    --max_eval_samples="${MAX_EVAL_SAMPLES:-64}")
fi

smolvla_repo_module lerobot.scripts.lerobot_train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=smolvla \
  --policy.pretrained_path="${PRETRAINED_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --policy.compile_model="${COMPILE_MODEL}" \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
  --policy.chunk_size="${ACTION_HORIZON}" \
  --policy.n_action_steps="${ACTION_HORIZON}" \
  --seed="${SEED}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --tolerance_s="${TOLERANCE_S}" "${split_args[@]}"

/home/s1/s1/Desktop/datasets/pick_place_brown_300