#!/usr/bin/env bash
# Daksha UI - Quick Setup & Test Script

set -euo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${UI_ROOT}/../../.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "Daksha VLA UI - Quick Setup"
echo "════════════════════════════════════════════════════════════════"

# Step 1: Select the global interpreter
echo ""
echo "1️⃣  Using global Python (no virtualenv)..."
cd "${UI_ROOT}"
PYTHON_BIN="${DAKSHA_PYTHON:-$(command -v python3)}"
echo "✓ Python interpreter: ${PYTHON_BIN}"

# Step 2: Install dependencies into the user site (--user leaves apt/system
# dist-packages in place and only shadows them for this user's python).
echo ""
echo "2️⃣  Installing UI dependencies into user site..."
"${PYTHON_BIN}" -m pip install --user -q -r requirements.txt
echo "✓ Dependencies installed"

# Step 3: Make scripts executable
echo ""
echo "3️⃣  Setting up scripts..."
chmod +x start_ui.sh
chmod +x ../smolvla/train_smolvla_local.sh
chmod +x ../act/train_act_local.sh
chmod +x ../diffusion/train_diffusion_local.sh
chmod +x ../pi_0_5/train_pi_0_5_local.sh
chmod +x inference_client.py
echo "✓ Scripts are executable"

# Step 4: Create necessary directories
echo ""
echo "4️⃣  Creating output directories..."
mkdir -p /tmp/daksha_logs
mkdir -p /tmp/daksha_uploads
mkdir -p "${REPO_ROOT}/outputs/daksha"
mkdir -p "${REPO_ROOT}/outputs/act_daksha"
mkdir -p "${REPO_ROOT}/outputs/diffusion_daksha"
mkdir -p "${REPO_ROOT}/outputs/pi_0_5_daksha"
mkdir -p "${REPO_ROOT}/outputs/smolvla_daksha"
echo "✓ Output directories created"

# Step 5: Verify configuration
echo ""
echo "5️⃣  Verifying configuration..."
if [[ ! -f model_configs.json ]]; then
    echo "❌ model_configs.json not found!"
    exit 1
fi

if python3 -c "import json; json.load(open('model_configs.json'))" 2>/dev/null; then
    echo "✓ Configuration file is valid"
else
    echo "❌ Configuration file is invalid JSON"
    exit 1
fi

# Step 6: Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the UI:"
echo "   cd ${UI_ROOT}"
echo "   bash start_ui.sh"
echo ""
echo "2. Open browser:"
echo "   http://localhost:7005"
echo ""
echo "3. Test inference (in another terminal):"
echo "   cd ${UI_ROOT}"
echo "   python3 inference_client.py --test"
echo ""
echo "════════════════════════════════════════════════════════════════"
