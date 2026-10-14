#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════════
# Daksha VLA UI — global dependency installer (replaces the per-container venv)
# ════════════════════════════════════════════════════════════════════════════════════
# Installs the full LeRobot/Daksha stack into the GLOBAL (user) site-packages so the UI,
# training, inference and dataset tooling all run under one system python3 — no venv.
#
# Why not just `pip install torch`? On Jetson Orin the working torch build (CUDA 12.6,
# sm_87 kernels) is NOT on PyPI and NOT on download.pytorch.org — the generic wheels
# omit sm_87. So torch/torchvision (and the dev-build diffusers) are COPIED out of the
# legacy container venv, and everything else is pip-installed --user from pinned wheels.
#
# Target: ~/.local/lib/python3.10/site-packages (user site). This shadows apt packages
# for this user's python only; it never overwrites /usr system packages and is undoable
# with `pip uninstall` / clearing ~/.local.
#
#   bash install_global_deps.sh
#   DAKSHA_PYTHON=/usr/bin/python3 bash install_global_deps.sh   # pick the interpreter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # <container> holding lerobot/ ui/ venv/
VENV_SP="${CONTAINER_ROOT}/venv/lib/python3.10/site-packages"
PYTHON_BIN="${DAKSHA_PYTHON:-$(command -v python3)}"

echo "Interpreter : ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
DEST="$("${PYTHON_BIN}" -c 'import site; print(site.getusersitepackages())')"
echo "User site   : ${DEST}"
echo "Legacy venv : ${VENV_SP}"
mkdir -p "${DEST}"

# ── 1. Copy the Jetson-only / non-PyPI packages straight from the legacy venv ────────
# torch + torchvision: Jetson sm_87 CUDA builds.  diffusers: a dev build not on PyPI.
COPY_PKGS=(torch torchvision diffusers)
if [[ -d "${VENV_SP}" ]]; then
    for pkg in torch torch-2.8.0.dist-info functorch torchgen \
               torchvision torchvision-0.23.0.dist-info \
               diffusers diffusers-0.35.0.dev0.dist-info; do
        if [[ -e "${VENV_SP}/${pkg}" && ! -e "${DEST}/${pkg}" ]]; then
            echo "  copy ${pkg}"
            cp -a "${VENV_SP}/${pkg}" "${DEST}/"
        fi
    done
else
    echo "WARNING: legacy venv not found at ${VENV_SP}."
    echo "         torch/torchvision/diffusers must be provided some other way"
    echo "         (a Jetson sm_87 torch wheel), or copied from another machine."
fi

# ── 2. pip-install the rest (pinned) into the user site ──────────────────────────────
# --no-deps keeps the exact, mutually-consistent closure from the venv and stops pip from
# dragging a generic torch wheel in as a transitive dependency over the copied one.
echo "Installing pinned requirements into user site..."
"${PYTHON_BIN}" -m pip install --user --no-warn-script-location --no-deps \
    -r "${SCRIPT_DIR}/requirements_global.txt"

# ── 3. Verify ────────────────────────────────────────────────────────────────────────
echo "Verifying the stack..."
PYTHONPATH="${CONTAINER_ROOT}/lerobot/src" "${PYTHON_BIN}" - <<'PY'
import torch, torchvision, lerobot, numpy, pandas, pyarrow, transformers, diffusers, cv2
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("lerobot OK | numpy", numpy.__version__, "| pandas", pandas.__version__)
PY

echo ""
echo "Done. The UI now runs under ${PYTHON_BIN} with the global stack."
echo "Known cosmetic conflicts (non-fatal): pyzed wants numpy>=2.0, colcon wants"
echo "setuptools<80, system scipy wants numpy<1.25 — all verified working at runtime."
