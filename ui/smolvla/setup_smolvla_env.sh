#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# After the restructure the LeRobot repo lives at <container>/lerobot (SCRIPT_DIR is
# <container>/ui/smolvla), so the repo is two levels up and then into lerobot/.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../lerobot" && pwd)"
cd "${REPO_ROOT}"

# Install the LeRobot stack and its `pi` extras into the global/user site-packages
# (--user) instead of a per-repo .venv. The editable install resolves lerobot from
# this repo's src/, matching the PYTHONPATH the UI uses. DAKSHA_PYTHON overrides the
# interpreter. NOTE: on Jetson, torch must be the sm_87 build already present globally;
# do not let this step pull a generic torch wheel over it.
PYTHON_BIN="${DAKSHA_PYTHON:-$(command -v python3)}"

"${PYTHON_BIN}" -m pip install --user -e ".[pi]"
"${PYTHON_BIN}" -m pip install --user msgpack pyzmq opencv-python

echo "smolvla dependencies installed globally for ${PYTHON_BIN} (repo: ${REPO_ROOT})"
