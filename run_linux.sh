#!/bin/bash
# ==============================================================================
# UAV AI Context-Switching Mission Control Console — Linux Launch Script
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================="
echo "UAV AI Context-Switching Mission Control Console"
echo "================================================================="

if [ -f "$DIR/model5/venv/bin/python3" ]; then
    PYTHON_BIN="$DIR/model5/venv/bin/python3"
elif [ -f "$DIR/venv/bin/python3" ]; then
    PYTHON_BIN="$DIR/venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

# Ensure streamlit is available
if ! "$PYTHON_BIN" -c "import streamlit" &> /dev/null; then
    echo "[SETUP] Installing missing dependencies from requirements.txt..."
    "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo "[INFO] Using Python environment: $PYTHON_BIN"
echo "[INFO] Starting Streamlit Mission Control GUI..."
"$PYTHON_BIN" -m streamlit run app/gui.py

