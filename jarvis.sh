#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/agents/main"
source "$SCRIPT_DIR/venv/bin/activate"

trap 'kill -9 $PID 2>/dev/null; exit 0' INT TERM

while true; do
    python -u "$SCRIPT_DIR/jarvis.py" "$@" &
    PID=$!
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Restarting Jarvis..."
    else
        exit $EXIT_CODE
    fi
done
