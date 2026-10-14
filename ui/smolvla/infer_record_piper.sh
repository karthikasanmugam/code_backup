#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

ROBOT_TYPE="${ROBOT_TYPE:-piper_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyUSB0}"
ROBOT_ID="${ROBOT_ID:-piper_arm_01}"
ROBOT_CAMERAS="${ROBOT_CAMERAS:-{world: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}, front_left: {type: opencv, index_or_path: 1, width: 672, height: 376, fps: 10}, front_right: {type: opencv, index_or_path: 2, width: 672, height: 376, fps: 10}, wrist: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 10}}}"

POLICY_PATH="${POLICY_PATH:-${SMOLVLA_DEFAULT_OUTPUT_ROOT}/smolvla_task1_full/checkpoints/last/pretrained_model}"
DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-false}"
EVAL_REPO_ID="${EVAL_REPO_ID:-local/eval_task1_full_smolvla}"
EVAL_ROOT="${EVAL_ROOT:-}"
TASK="${TASK:-${SMOLVLA_DEFAULT_TASK}}"
NUM_EPISODES="${NUM_EPISODES:-10}"
FPS="${FPS:-10}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

if [[ -z "${EVAL_ROOT}" ]]; then
  echo "Please set EVAL_ROOT for where evaluation dataset should be saved."
  exit 1
fi

smolvla_repo_module lerobot.scripts.lerobot_record \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${EVAL_REPO_ID}" \
  --dataset.root="${EVAL_ROOT}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${FPS}" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  --policy.use_amp="${USE_AMP}"
