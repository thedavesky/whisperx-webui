#!/bin/bash
set -e

source venv/bin/activate
exec python app.py "$@"
