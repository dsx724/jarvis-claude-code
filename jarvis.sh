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

        # Preflight checks before restart
        if python -c "import py_compile; py_compile.compile('$SCRIPT_DIR/test_preflight.py', doraise=True)" 2>/dev/null; then
            echo "Running preflight checks..."
            if timeout 120 python "$SCRIPT_DIR/test_preflight.py"; then
                echo "Preflight passed, restarting."
            else
                echo "Preflight FAILED. Reverting last commit..."
                if git -C "$SCRIPT_DIR" revert --no-edit HEAD; then
                    echo "Reverted HEAD. Retrying preflight..."
                    if timeout 120 python "$SCRIPT_DIR/test_preflight.py"; then
                        echo "Preflight passed after revert, restarting."
                    else
                        echo "Preflight still failing after revert. Starting anyway."
                    fi
                else
                    echo "git revert failed, falling back to checkout..."
                    git -C "$SCRIPT_DIR" checkout HEAD~1 -- jarvis.py config/config.py
                    echo "Restored jarvis.py and config/config.py from HEAD~1."
                fi
            fi
        else
            echo "WARNING: test_preflight.py has syntax errors, skipping preflight."
        fi
    else
        exit $EXIT_CODE
    fi
done
