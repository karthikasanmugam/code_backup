#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for examples/daksha/inference/*.sh. Mirrors
# examples/daksha/smolvla/common.sh's python-resolution logic (same repo, same venv
# convention) but points at examples/daksha/serve_policy_zmq.py -- the generic
# multi-policy server (ACT, Diffusion, SmolVLA, ...) rather than smolvla's own script.

INFERENCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_REPO_ROOT="$(cd "${INFERENCE_SCRIPT_DIR}/../../.." && pwd)"
INFERENCE_DAKSHA_ROOT="$(cd "${INFERENCE_SCRIPT_DIR}/.." && pwd)"

# repo_python: runs the checkpoint-loading policy server. Needs lerobot's real deps
# (torch, ...), which after the global-libraries migration live in the global/user
# site-packages, so it runs under the system python3 (DAKSHA_PYTHON overrides).
inference_repo_python() {
  local repo_python="${DAKSHA_PYTHON:-$(command -v python3 || true)}"
  if [[ -n "${repo_python}" && -x "${repo_python}" ]]; then
    PYTHONPATH="${INFERENCE_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${repo_python}" "$@"
    return
  fi

  echo "No python3 interpreter found for the LeRobot stack." >&2
  echo "Install the dependencies globally, or set DAKSHA_PYTHON to a python executable that has them." >&2
  exit 1
}

# ros_python: runs the ROS2 bridge. It only needs rclpy/cv2/msgpack/pyzmq/numpy -- never
# lerobot itself -- so it deliberately runs under the system/ROS interpreter, not the
# repo's lerobot venv (which does not have rclpy installed).
inference_ros_python() {
  local ros_python="${ROS_PYTHON_BIN:-/usr/bin/python3}"
  local ros_setup="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
  if [[ -f "${ros_setup}" ]]; then
    set +u
    source "${ros_setup}"
    set -u
  fi
  "${ros_python}" "$@"
}

# resolve_checkpoint <path>: prints the actual pretrained_model dir for a checkpoint
# path that may be either a training run's output_dir or already a pretrained_model
# dir. See resolve_checkpoint.py for the resolution rules (mirrors app.py's
# resolve_pretrained_model_dir so the UI's checkpoint list and this launcher agree).
inference_resolve_checkpoint() {
  inference_repo_python "${INFERENCE_DAKSHA_ROOT}/inference/resolve_checkpoint.py" "$1"
}
