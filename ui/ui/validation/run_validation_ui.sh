#!/usr/bin/env bash
# Dataset validation UI — type a dataset path, press Run, read which episodes
# are wrong and why, with joint traces and the videos for any episode.
#
#   bash /home/s1/.ihub/vla/lerobot_2/ui/ui/validation/run_validation_ui.sh
#   PORT=9000 bash .../run_validation_ui.sh
#   TOKEN=mysecret bash .../run_validation_ui.sh      # require ?token=mysecret
#   HOST=127.0.0.1 bash .../run_validation_ui.sh      # this machine only
#   DATASET_PATH=/data/ds bash .../run_validation_ui.sh
#
# It binds every interface by default so you can open it from another machine;
# the script prints the exact URLs. Anyone who can reach the port can read any
# dataset under ALLOW_ROOT, so set TOKEN on a shared network.
set -euo pipefail

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${PORT:-8124}"
HOST="${HOST:-0.0.0.0}"
TOKEN="${TOKEN:-}"
ALLOW_ROOT="${ALLOW_ROOT:-$HOME}"
DATASET_PATH="${DATASET_PATH:-/home/s1/Desktop/datasets/pick_place_brown_300}"

# pyarrow/numpy now live in the global/user site-packages; run under the global
# Python (DAKSHA_VALIDATION_PYTHON/DAKSHA_PYTHON override the interpreter).
PYTHON_BIN="${DAKSHA_VALIDATION_PYTHON:-${DAKSHA_PYTHON:-$(command -v python3)}}"

# Free the port if a previous UI server is still holding it.
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

exec "$PYTHON_BIN" "$UI_DIR/validation_ui.py" \
  --host "$HOST" --port "$PORT" \
  --allow-root "$ALLOW_ROOT" \
  --dataset "$DATASET_PATH" \
  ${TOKEN:+--token "$TOKEN"}
