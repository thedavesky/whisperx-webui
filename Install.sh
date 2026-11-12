#!/bin/bash

WEBUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WEBUI_DIR" || exit 1

PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 was not found. Install Python 3.12, then run this script again."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv || {
        echo ""
        echo "Virtual environment creation failed. Fix the error above and run the script again."
        exit 1
    }
fi

VENV_PYTHON="$WEBUI_DIR/venv/bin/python"
VENV_PIP="$WEBUI_DIR/venv/bin/pip"

"$VENV_PYTHON" -m pip install -U pip

"$VENV_PIP" install -U -r "$WEBUI_DIR/requirements.txt" && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    exit 1
}
