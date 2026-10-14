#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend_v3"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
# The validator now runs under the global Python (no virtualenv); DAKSHA_VALIDATION_PYTHON
# or DAKSHA_PYTHON override the interpreter.
PYTHON_BIN="${DAKSHA_VALIDATION_PYTHON:-${DAKSHA_PYTHON:-$(command -v python3)}}"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5500"

LOG_DIR="${PROJECT_ROOT}/logs"
BACKEND_LOG="${LOG_DIR}/backend.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo
    echo "Stopping validator services..."

    if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        kill "${BACKEND_PID}" 2>/dev/null || true
    fi

    if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        kill "${FRONTEND_PID}" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    echo "Validator services stopped."
}

trap cleanup INT TERM EXIT

if [[ ! -d "${BACKEND_DIR}" ]]; then
    echo "Backend directory not found: ${BACKEND_DIR}" >&2
    exit 1
fi

if [[ ! -d "${FRONTEND_DIR}" ]]; then
    echo "Frontend directory not found: ${FRONTEND_DIR}" >&2
    exit 1
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "No usable python3 found (set DAKSHA_VALIDATION_PYTHON)." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Starting backend (python: ${PYTHON_BIN})..."
cd "${BACKEND_DIR}"

# Episode deletion imports lerobot.datasets.dataset_tools. lerobot is not pip-installed
# for the global Python, so make the in-repo source checkout importable. Override with
# DAKSHA_LEROBOT_SRC if the checkout lives elsewhere.
LEROBOT_SRC="${DAKSHA_LEROBOT_SRC:-/home/s1/.ihub/vla/lerobot_2/lerobot/src}"
if [[ ! -d "${LEROBOT_SRC}/lerobot" ]]; then
    echo "WARNING: source not found at ${LEROBOT_SRC}; episode deletion will fail." >&2
fi

PYTHONPATH="src:${LEROBOT_SRC}" "${PYTHON_BIN}" -m uvicorn lerobot_validator.api.app:app \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    --reload \
    >"${BACKEND_LOG}" 2>&1 &

BACKEND_PID=$!

echo "Starting frontend..."
cd "${FRONTEND_DIR}"

"${PYTHON_BIN}" -m http.server "${FRONTEND_PORT}" \
    --bind "${FRONTEND_HOST}" \
    >"${FRONTEND_LOG}" 2>&1 &

FRONTEND_PID=$!

sleep 2

if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend failed to start. Check:"
    echo "  ${BACKEND_LOG}"
    exit 1
fi

if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "Frontend failed to start. Check:"
    echo "  ${FRONTEND_LOG}"
    exit 1
fi

echo
echo "=============================================="
echo "  Dataset Validator"
echo "=============================================="
echo " Backend : http://${BACKEND_HOST}:${BACKEND_PORT}"
echo " Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo " API     : http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"
echo " Logs    : ${LOG_DIR}"
echo
echo " Open:"
echo " http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo
echo " Press Ctrl+C to stop both services."
echo "=============================================="
echo

while true; do
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "Backend stopped unexpectedly. Check: ${BACKEND_LOG}" >&2
        exit 1
    fi

    if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "Frontend stopped unexpectedly. Check: ${FRONTEND_LOG}" >&2
        exit 1
    fi

    sleep 2
done