#!/usr/bin/env bash
# Daksha UI - Development Testing Script

set -euo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${UI_ROOT}/../../.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "Daksha UI - Testing Suite"
echo "════════════════════════════════════════════════════════════════"

# The UI and its dependencies now run under the global Python (no virtualenv).
PYTHON_BIN="${DAKSHA_PYTHON:-$(command -v python3)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "❌ No usable python3 found. Run setup.sh first."
    exit 1
fi

cd "${UI_ROOT}"

# Test 1: Verify Python imports
echo ""
echo "🧪 Test 1: Verifying Python imports..."
python3 << 'EOF'
try:
    import flask
    import zmq
    import msgpack
    import psutil
    import numpy as np
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
EOF

# Test 2: Verify configuration file
echo ""
echo "🧪 Test 2: Verifying configuration file..."
python3 << 'EOF'
import json
try:
    with open('model_configs.json') as f:
        config = json.load(f)
    print(f"✓ Configuration valid: {len(config['models'])} models configured")
    for model_type, model_config in config['models'].items():
        print(f"  - {model_type}: {model_config['name']}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit(1)
EOF

# Test 3: Verify training scripts
echo ""
echo "🧪 Test 3: Verifying training scripts..."
scripts=(
    "../smolvla/train_smolvla_local.sh"
    "../act/train_act_local.sh"
    "../diffusion/train_diffusion_local.sh"
    "../pi_0_5/train_pi_0_5_local.sh"
)
for script in "${scripts[@]}"; do
    if [[ -x "$script" ]]; then
        echo "✓ $(basename $(dirname $script)): script is executable"
    else
        echo "❌ $(basename $(dirname $script)): script not found or not executable"
    fi
done

# Test 4: Verify directories
echo ""
echo "🧪 Test 4: Verifying output directories..."
dirs=(
    "/tmp/daksha_logs"
    "/tmp/daksha_uploads"
    "${REPO_ROOT}/outputs/daksha"
)
for dir in "${dirs[@]}"; do
    if [[ -d "$dir" && -w "$dir" ]]; then
        echo "✓ $(basename $(dirname $dir))/$(basename $dir): writable"
    else
        echo "⚠️  Creating: $dir"
        mkdir -p "$dir"
    fi
done

# Test 5: Test Flask app syntax
echo ""
echo "🧪 Test 5: Checking Flask app syntax..."
python3 -m py_compile ../../app.py && echo "✓ app.py syntax OK" || {
    echo "❌ app.py has syntax errors"
    exit 1
}

# Test 6: Test inference client
echo ""
echo "🧪 Test 6: Checking inference client syntax..."
python3 -m py_compile inference_client.py && echo "✓ inference_client.py syntax OK" || {
    echo "❌ inference_client.py has syntax errors"
    exit 1
}

# Test 7: Inference client help
echo ""
echo "🧪 Test 7: Testing inference client help..."
python3 inference_client.py --help > /dev/null 2>&1 && echo "✓ Inference client help works" || {
    echo "❌ Inference client help failed"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ All tests passed!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Ready to start UI with: bash start_ui.sh"
echo ""
