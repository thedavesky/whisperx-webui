#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

VENV_PYTHON=venv/bin/python
VENV_PIP=venv/bin/pip

"$VENV_PYTHON" -m pip install -U pip

"$VENV_PIP" install -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    exit 1
}
