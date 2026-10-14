#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

POLICY_PATH="${POLICY_PATH:-${SMOLVLA_DEFAULT_OUTPUT_ROOT}/smolvla_task1_full/checkpoints/last/pretrained_model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
ROBOT_TYPE="${ROBOT_TYPE:-leader_follower_6dof_4camera}"
TASK_DEFAULT="${TASK_DEFAULT:-${SMOLVLA_DEFAULT_TASK}}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"

smolvla_repo_python examples/daksha/smolvla/serve_smolvla_zmq.py \
  --policy-path "${POLICY_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --robot-type "${ROBOT_TYPE}" \
  --task-default "${TASK_DEFAULT}" \
  --action-horizon "${ACTION_HORIZON}"
