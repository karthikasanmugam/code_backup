#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════════════
# Daksha VLA UI - Startup Script
# Complete setup and launch for training and inference UI
# ════════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_ROOT="${SCRIPT_DIR}"
# After the restructure this script lives at <container>/ui/ui/start_ui.sh, so the
# container (which holds venv/, ui/, lerobot/) is two levels up, not three.
REPO_ROOT="$(cd "${UI_ROOT}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ════════════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        print_success "$1 found"
        return 0
    else
        print_error "$1 not found"
        return 1
    fi
}

check_python_module() {
    if "${PYTHON_BIN}" -c "import $1" 2>/dev/null; then
        print_success "Python module: $1"
        return 0
    else
        print_error "Python module: $1 not installed"
        return 1
    fi
}

find_free_port() {
    local start_port="${1:-8104}"
    local candidate="${start_port}"

    while (( candidate <= 65535 )); do
        if python3 - "$candidate" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    s.close()
PY
        then
            echo "$candidate"
            return 0
        fi
        candidate=$((candidate + 1))
    done

    return 1
}

# ════════════════════════════════════════════════════════════════════════════════════
# MAIN SETUP
# ════════════════════════════════════════════════════════════════════════════════════

print_header "Daksha VLA Training & Inference UI - Setup"

# Check Python installation
print_info "Checking system dependencies..."
check_command python3 || {
    print_error "Python 3 is required"
    exit 1
}

# The torch/lerobot stack and the UI dependencies now live in the global (user/system)
# site-packages rather than a per-container venv, so the UI runs directly under the
# system Python. DAKSHA_PYTHON overrides the interpreter for non-standard setups.
PYTHON_BIN="${DAKSHA_PYTHON:-$(command -v python3)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    print_error "No usable python3 found (set DAKSHA_PYTHON to a python executable)."
    exit 1
fi
export DAKSHA_PYTHON="${PYTHON_BIN}"
print_success "Using global Python interpreter: ${PYTHON_BIN}"

# Install any missing UI packages into the user site (--user keeps system/apt
# dist-packages untouched; it only shadows them for this user's python). Pins live in
# requirements.txt; here we only add what's missing so a working stack is never changed.
declare -A UI_PIP_NAMES=([flask]=flask [flask_cors]=flask-cors [psutil]=psutil [zmq]=pyzmq [msgpack]=msgpack)
MISSING_PKGS=()
for mod in "${!UI_PIP_NAMES[@]}"; do
    "${PYTHON_BIN}" -c "import ${mod}" 2>/dev/null || MISSING_PKGS+=("${UI_PIP_NAMES[$mod]}")
done
if (( ${#MISSING_PKGS[@]} )); then
    print_info "Installing missing packages into user site: ${MISSING_PKGS[*]}"
    "${PYTHON_BIN}" -m pip install --user "${MISSING_PKGS[@]}"
fi

# Verify installations against the interpreter the UI will actually run under.
print_info "Verifying installations..."
check_python_module flask || exit 1
check_python_module zmq || exit 1
check_python_module msgpack || exit 1

# ════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════

print_header "Configuration Setup"

# Check for configuration file
if [[ ! -f "${UI_ROOT}/model_configs.json" ]]; then
    print_error "model_configs.json not found"
    exit 1
fi
print_success "Configuration file found"

# Create output directories if they don't exist
print_info "Creating output directories..."
mkdir -p /tmp/daksha_logs
mkdir -p /tmp/daksha_uploads
mkdir -p "${REPO_ROOT}/outputs/daksha"
print_success "Output directories created"

# ════════════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════════════

print_header "Daksha VLA UI - Starting Server"

# Default configuration
PORT="${DAKSHA_PORT:-$(find_free_port 8104)}"
HOST="${DAKSHA_HOST:-0.0.0.0}"
DEBUG="${DAKSHA_DEBUG:-false}"
# DATASET_ROOT's immediate subdirectories are listed as selectable datasets in the UI
# (see app.py's list_datasets()); this is where recorded Daksha datasets live on this
# machine (pick_place_brown_300, pick_box_1, pick_box_2, etc.).
export DAKSHA_DATASET_ROOT="${DAKSHA_DATASET_ROOT:-/home/s1/Desktop/datasets}"

if [[ -z "${DAKSHA_PORT:-}" ]]; then
    print_info "Port 8104 was in use; switching to free port ${PORT}"
fi

print_info "Configuration:"
print_info "  Host: $HOST"
print_info "  Port: $PORT"
print_info "  Debug: $DEBUG"
print_info "  Dataset root: ${DAKSHA_DATASET_ROOT}"

print_info ""
print_info "UI will be available at: http://localhost:$PORT"
print_info "Also reachable on the machine network at: http://${HOST}:$PORT"
print_info "Press Ctrl+C to stop the server"
print_info ""

# Export environment variables
export FLASK_APP="${REPO_ROOT}/app.py"
export FLASK_ENV="${DEBUG}"
export PYTHONUNBUFFERED=1

# Start the Flask application using the global interpreter.
cd "${UI_ROOT}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/app.py" --host="$HOST" --port="$PORT"
