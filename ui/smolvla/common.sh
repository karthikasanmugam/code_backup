#!/usr/bin/env bash
set -euo pipefail

SMOLVLA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# After the restructure this lives at <container>/ui/smolvla/common.sh. The LeRobot repo
# (src/, outputs/, pyproject.toml) is at <container>/lerobot; the Python environment is at
# <container>/venv. A compatibility symlink <container>/lerobot/examples/daksha -> ../../ui
# keeps every "cd into the repo then run examples/daksha/..." reference in the launchers
# resolving to the real ui/ folder, so those relative paths did not need rewriting.
SMOLVLA_CONTAINER_ROOT="$(cd "${SMOLVLA_SCRIPT_DIR}/../.." && pwd)"
SMOLVLA_REPO_ROOT="$(cd "${SMOLVLA_CONTAINER_ROOT}/lerobot" && pwd)"

SMOLVLA_DEFAULT_DATASET_REPO_ID="local/pick_place_brown"
SMOLVLA_DEFAULT_DATASET_ROOT="/home/s1/Desktop/datasets/pick_place_brown_300"
SMOLVLA_DEFAULT_PRETRAINED_PATH="lerobot/smolvla_base"
SMOLVLA_DEFAULT_OUTPUT_ROOT="outputs/smolvla"
SMOLVLA_DEFAULT_TASK="task1"

smolvla_plain_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  echo "No python interpreter found in PATH." >&2
  exit 1
}

smolvla_repo_python() {
  if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    if [[ ! -x "${SMOLVLA_PYTHON}" ]]; then
      echo "SMOLVLA_PYTHON is not executable: ${SMOLVLA_PYTHON}" >&2
      exit 1
    fi
    PYTHONPATH="${SMOLVLA_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${SMOLVLA_PYTHON}" "$@"
    return
  fi

  # After the global-libraries migration the torch/lerobot stack (incl. the Jetson
  # sm_87 torch) lives in the global/user site-packages, so fall back to the system
  # python3. The UI still passes SMOLVLA_PYTHON for ACT/SmolVLA; diffusion reaches
  # this fallback. DAKSHA_PYTHON (if exported) takes precedence over a bare python3.
  local fallback_python="${DAKSHA_PYTHON:-$(command -v python3 || true)}"
  if [[ -n "${fallback_python}" && -x "${fallback_python}" ]]; then
    PYTHONPATH="${SMOLVLA_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${fallback_python}" "$@"
    return
  fi

  echo "No python3 interpreter found for the LeRobot stack." >&2
  echo "Install the dependencies globally, or set SMOLVLA_PYTHON/DAKSHA_PYTHON to a python executable that has them." >&2
  exit 1
}

smolvla_repo_module() {
  local module="$1"
  shift
  smolvla_repo_python -m "${module}" "$@"
}

smolvla_maybe_ulimit() {
  if [[ -n "${ULIMIT_NOFILE:-}" ]]; then
    ulimit -n "${ULIMIT_NOFILE}" || true
  fi
}
