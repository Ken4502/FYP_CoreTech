#!/usr/bin/env bash
set -e

echo "======================================================================="
echo "         FYP CoreTech: One-Click Plug & Play AI Launcher"
echo "======================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 was not found on your system PATH!"
    echo "Please install Python 3.10+."
    exit 1
fi

# 2. Check and Pull Git LFS Model Weights
if command -v git &> /dev/null; then
    echo "[*] Checking model weights with Git LFS..."
    git lfs pull || true
    echo ""
fi

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[*] First-time setup detected: Creating virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ""
else
    source .venv/bin/activate
fi

# 4. Launch Services
echo "[*] Launching FloraNet-250 (Port 8000) and PhytoCare Ultra (Port 8001)..."
python launch.py "$@"

