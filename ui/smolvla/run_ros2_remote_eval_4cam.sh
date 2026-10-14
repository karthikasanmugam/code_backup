#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${SMOLVLA_REPO_ROOT}"

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
CONTROL_HZ="${CONTROL_HZ:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
LANG_INSTRUCTION="${LANG_INSTRUCTION:-${SMOLVLA_DEFAULT_TASK}}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_PYTHON_BIN="${ROS_PYTHON_BIN:-/usr/bin/python3}"

if [[ -f "${ROS_SETUP}" ]]; then
  set +u
  source "${ROS_SETUP}"
  set -u
fi

"${ROS_PYTHON_BIN}" examples/daksha/smolvla/eval_piper_ros2_remote_4cam.py \
  --policy-host "${POLICY_HOST}" \
  --policy-port "${POLICY_PORT}" \
  --control-hz "${CONTROL_HZ}" \
  --action-horizon "${ACTION_HORIZON}" \
  --joint-state-topic /joint_states_right \
  --cmd-topic /joint_ctrl_cmd_right \
  --world-topic /world/camera/image_raw/compressed \
  --front-left-topic /front/left/image_raw/compressed \
  --front-right-topic /front/right/image_raw/compressed \
  --wrist-topic /waveshare/cam1/image_raw/compressed \
  --world-camera-key world \
  --front-left-camera-key front_left \
  --front-right-camera-key front_right \
  --wrist-camera-key wrist \
  --lang-instruction "${LANG_INSTRUCTION}"
