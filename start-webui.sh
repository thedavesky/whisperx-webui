#!/bin/bash
set -e

WEBUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WEBUI_DIR"

if [ ! -f "venv/bin/activate" ]; then
    echo "No virtualenv found. Run Install.sh first."
    exit 1
fi

source venv/bin/activate
exec python app.py "$@"
