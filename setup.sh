#!/bin/bash
# Jarvis setup script — installs system deps, creates venv, installs Python packages

set -e

# System dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libpulse0 alsa-utils

# Python venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install openwakeword faster-whisper numpy piper-tts

echo ""
echo "Setup complete. Run with:"
echo "  cd $SCRIPT_DIR && source venv/bin/activate && python jarvis.py"
