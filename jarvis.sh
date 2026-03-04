#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/venv/bin/activate"

while true; do
    python -u "$SCRIPT_DIR/jarvis.py" "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Restarting Jarvis..."
    else
        exit $EXIT_CODE
    fi
done
