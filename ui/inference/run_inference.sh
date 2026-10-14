#!/usr/bin/env bash
set -euo pipefail

# Starts inference for a Daksha B1 checkpoint end to end: resolve the checkpoint,
# load it into examples/daksha/serve_policy_zmq.py (works for ACT, Diffusion,
# SmolVLA, ... -- the policy type comes from the checkpoint's own config), then start
# the ROS2 bridge. The bridge does not begin commanding the arm until it has
# confirmed the checkpoint actually finished loading (eval_daksha.py's
# wait_until_ready), so there is no "bridge starts before the model is ready" race
# even though the server is only ever started in the background here.
#
# Usage:
#   POLICY_PATH=/path/to/run_or_pretrained_model \
#   JOINT_MIN_CSV=... JOINT_MAX_CSV=... MAX_STEP_DELTA_CSV=... \
#     ./run_inference.sh [extra args passed through to eval_daksha.py]
#
# POLICY_PATH may be either a training run's output_dir (as listed by the Daksha UI's
# /api/checkpoints) or an already-resolved pretrained_model dir -- see
# resolve_checkpoint.py. JOINT_MIN_CSV / JOINT_MAX_CSV / MAX_STEP_DELTA_CSV have no
# default; see eval_daksha.py's module docstring for how to derive them for
# this checkpoint's training dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${INFERENCE_REPO_ROOT}"

: "${POLICY_PATH:?Set POLICY_PATH to a checkpoint run dir or pretrained_model dir}"
: "${JOINT_MIN_CSV:?Set JOINT_MIN_CSV -- see eval_daksha.py docstring to derive it}"
: "${JOINT_MAX_CSV:?Set JOINT_MAX_CSV -- see eval_daksha.py docstring to derive it}"
: "${MAX_STEP_DELTA_CSV:?Set MAX_STEP_DELTA_CSV -- see eval_daksha.py docstring to derive it}"

HOST="${POLICY_HOST:-127.0.0.1}"
PORT="${POLICY_PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
CONTROL_HZ="${CONTROL_HZ:-30}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
LANG_INSTRUCTION="${LANG_INSTRUCTION:-task1}"
SERVER_LOG="${SERVER_LOG:-/tmp/daksha_inference_server.log}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-120}"

echo "[run_inference] Resolving checkpoint: ${POLICY_PATH}"
RESOLVED_POLICY_PATH="$(inference_resolve_checkpoint "${POLICY_PATH}")"
echo "[run_inference] Using pretrained_model dir: ${RESOLVED_POLICY_PATH}"

echo "[run_inference] Starting policy server (loading checkpoint onto ${DEVICE})..."
echo "[run_inference]   log: ${SERVER_LOG}"
inference_repo_python "${INFERENCE_DAKSHA_ROOT}/serve_policy_zmq.py" \
  --policy-path "${RESOLVED_POLICY_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --task-default "${LANG_INSTRUCTION}" \
  --action-horizon "${ACTION_HORIZON}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[run_inference] Stopping policy server (pid ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[run_inference] Policy server exited immediately -- see ${SERVER_LOG}" >&2
  exit 1
fi

# The bridge itself patiently waits on ping() until the checkpoint has actually
# finished loading (see wait_until_ready in eval_daksha.py) -- it is safe
# to start it now even though the server above may still be mid-load.
echo "[run_inference] Starting ROS2 bridge (will wait for the server to be ready)..."
inference_ros_python "${INFERENCE_DAKSHA_ROOT}/inference/eval_daksha.py" \
  --policy-host "${HOST}" \
  --policy-port "${PORT}" \
  --server-wait-timeout-s "${SERVER_WAIT_TIMEOUT_S}" \
  --control-hz "${CONTROL_HZ}" \
  --action-horizon "${ACTION_HORIZON}" \
  --lang-instruction "${LANG_INSTRUCTION}" \
  --joint-min-csv="${JOINT_MIN_CSV}" \
  --joint-max-csv="${JOINT_MAX_CSV}" \
  --max-step-delta-csv="${MAX_STEP_DELTA_CSV}" \
  "$@"
