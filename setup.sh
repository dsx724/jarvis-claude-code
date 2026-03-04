#!/bin/bash
# Jarvis setup script — installs system deps, creates venv, installs Python packages.
# Designed to be idempotent and fast: skips anything already installed.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- System dependencies ---
SYSTEM_PKGS=(libpulse0 alsa-utils)
missing_pkgs=()
for pkg in "${SYSTEM_PKGS[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        missing_pkgs+=("$pkg")
    fi
done

if [ ${#missing_pkgs[@]} -gt 0 ]; then
    echo "Installing system packages: ${missing_pkgs[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_pkgs[@]}"
fi

# --- Python venv ---
if [ ! -d venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# --- Python packages ---
PY_PACKAGES=(openwakeword faster_whisper numpy piper)
PIP_NAMES=(openwakeword faster-whisper numpy piper-tts)
missing_pip=()
for i in "${!PY_PACKAGES[@]}"; do
    if ! python -c "import ${PY_PACKAGES[$i]}" &>/dev/null; then
        missing_pip+=("${PIP_NAMES[$i]}")
    fi
done

if [ ${#missing_pip[@]} -gt 0 ]; then
    echo "Installing Python packages: ${missing_pip[*]}"
    pip install --upgrade pip
    pip install "${missing_pip[@]}"
fi

echo "Setup OK — all dependencies present."
