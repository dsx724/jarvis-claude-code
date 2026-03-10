#!/bin/bash
# Jarvis setup script — installs system deps, creates venv, installs Python packages.
# Designed to be idempotent and fast: skips anything already installed.
#
# GPU acceleration packages are installed automatically based on detected hardware:
#   Intel GPU  → optimum-intel[openvino], transformers (OpenVINO STT/TTS acceleration)
#   NVIDIA GPU → (future: CUDA support)
#   CPU-only   → OpenVINO CPU backend if x86_64/arm64 (still faster than raw CPU for STT)
#
# Override with: --force-openvino, --force-cuda, --cpu-only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FORCE_OPENVINO=0
FORCE_CUDA=0
CPU_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --force-openvino) FORCE_OPENVINO=1 ;;
        --force-cuda)     FORCE_CUDA=1 ;;
        --cpu-only)       CPU_ONLY=1 ;;
    esac
done

# --- System dependencies ---
SYSTEM_PKGS=(libpulse0 alsa-utils ffmpeg)
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

# --- Core Python packages ---
PY_PACKAGES=(openwakeword faster_whisper numpy piper telegram)
PIP_NAMES=(openwakeword faster-whisper numpy piper-tts python-telegram-bot)
missing_pip=()
for i in "${!PY_PACKAGES[@]}"; do
    if ! python -c "import ${PY_PACKAGES[$i]}" &>/dev/null; then
        missing_pip+=("${PIP_NAMES[$i]}")
    fi
done

if [ ${#missing_pip[@]} -gt 0 ]; then
    # Warn about model licenses bundled with packages
    for pkg in "${missing_pip[@]}"; do
        case "$pkg" in
            openwakeword)
                echo "NOTE: openwakeword includes pre-trained models licensed CC BY-NC-SA 4.0"
                echo "      (non-commercial use only). See https://github.com/dscripka/openWakeWord"
                ;;
            piper-tts)
                echo "NOTE: piper-tts is licensed GPL-3.0-or-later."
                echo "      Piper voice models have their own licenses — some are CC BY-NC-SA 4.0"
                echo "      (non-commercial). Check voice license before redistributing."
                echo "      See https://huggingface.co/rhasspy/piper-voices"
                ;;
        esac
    done
    echo "Installing Python packages: ${missing_pip[*]}"
    pip install --upgrade pip
    pip install "${missing_pip[@]}"
fi

# --- GPU detection & acceleration packages ---
if [ "$CPU_ONLY" -eq 0 ]; then
    # Detect hardware
    HAS_INTEL_GPU=0
    HAS_NVIDIA_GPU=0

    if command -v lspci &>/dev/null; then
        if lspci 2>/dev/null | grep -iE '(vga|3d|display)' | grep -iq 'intel'; then
            HAS_INTEL_GPU=1
        fi
        if lspci 2>/dev/null | grep -iE '(vga|3d|display)' | grep -iq 'nvidia'; then
            HAS_NVIDIA_GPU=1
        fi
    fi

    # OpenVINO: install if Intel GPU detected, or if on x86_64/arm64 (CPU acceleration),
    # or if forced via --force-openvino
    INSTALL_OPENVINO=0
    ARCH=$(uname -m)
    if [ "$FORCE_OPENVINO" -eq 1 ]; then
        INSTALL_OPENVINO=1
        echo "GPU: OpenVINO forced via --force-openvino"
    elif [ "$HAS_INTEL_GPU" -eq 1 ]; then
        INSTALL_OPENVINO=1
        echo "GPU: Intel GPU detected — installing OpenVINO for GPU+CPU acceleration"
    elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then
        INSTALL_OPENVINO=1
        echo "GPU: No Intel GPU, but $ARCH CPU supports OpenVINO CPU acceleration"
    fi

    if [ "$INSTALL_OPENVINO" -eq 1 ]; then
        ov_missing=()
        python -c "from optimum.intel import OVModelForSpeechSeq2Seq" &>/dev/null || ov_missing+=("optimum-intel[openvino]")
        python -c "import transformers" &>/dev/null || ov_missing+=("transformers")
        if [ ${#ov_missing[@]} -gt 0 ]; then
            echo "Installing OpenVINO packages: ${ov_missing[*]}"
            pip install "${ov_missing[@]}"
        fi
    fi

    # CUDA: future support
    if [ "$FORCE_CUDA" -eq 1 ] || [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
        echo "GPU: NVIDIA GPU detected — CUDA STT support not yet implemented"
        # Future: pip install faster-whisper[cuda] or similar
    fi
else
    echo "GPU: Skipped (--cpu-only)"
fi

echo "Setup OK — all dependencies present."
