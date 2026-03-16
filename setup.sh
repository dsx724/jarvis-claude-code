#!/bin/bash
# Jarvis setup script -- installs system deps, creates venv, installs Python packages.
# Designed to be idempotent and fast: skips anything already installed.
#
# GPU acceleration packages are installed automatically based on detected hardware:
#   WSL2 + /dev/dxg  -> onnxruntime-directml (Qualcomm Adreno, AMD, NVIDIA via D3D12)
#   Intel GPU        -> optimum-intel[openvino], transformers (OpenVINO STT/TTS acceleration)
#   NVIDIA GPU       -> (future: CUDA support)
#   CPU-only         -> OpenVINO CPU backend if x86_64 (still faster than raw CPU for STT)
#
# Note: onnxruntime and onnxruntime-directml conflict -- only one can be installed.
# On WSL2 with GPU passthrough, onnxruntime-directml is preferred over onnxruntime.
#
# Override with: --force-directml, --force-openvino, --force-cuda, --cpu-only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FORCE_DIRECTML=0
FORCE_OPENVINO=0
FORCE_CUDA=0
CPU_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --force-directml) FORCE_DIRECTML=1 ;;
        --force-openvino) FORCE_OPENVINO=1 ;;
        --force-cuda)     FORCE_CUDA=1 ;;
        --cpu-only)       CPU_ONLY=1 ;;
    esac
done

# --- System dependencies ---
SYSTEM_PKGS=(libpulse0 alsa-utils ffmpeg inotify-tools)
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

# --- Python venv (per-architecture to avoid binary mismatch on shared filesystems) ---
VENV_DIR="venv-$(uname -m)"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# --- Core Python packages ---
PY_PACKAGES=(openwakeword faster_whisper numpy piper telegram silero_vad transformers)
PIP_NAMES=(openwakeword faster-whisper numpy piper-tts python-telegram-bot silero-vad transformers)
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
GPU_STAMP="$SCRIPT_DIR/.gpu-setup-stamp"

if [ "$CPU_ONLY" -eq 1 ]; then
    echo "GPU: Skipped (--cpu-only)"
else
    ARCH=$(uname -m)

    # Detect environment
    IS_WSL=0
    grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1

    HAS_DXG=0
    [ -e /dev/dxg ] && HAS_DXG=1

    HAS_INTEL_GPU=0
    HAS_NVIDIA_GPU=0
    if command -v lspci &>/dev/null; then
        lspci 2>/dev/null | grep -iE '(vga|3d|display)' | grep -iq 'intel'  && HAS_INTEL_GPU=1
        lspci 2>/dev/null | grep -iE '(vga|3d|display)' | grep -iq 'nvidia' && HAS_NVIDIA_GPU=1
    fi

    # Build a string describing the current GPU environment
    GPU_ENV="wsl=${IS_WSL},dxg=${HAS_DXG},intel=${HAS_INTEL_GPU},nvidia=${HAS_NVIDIA_GPU},arch=${ARCH},force_dm=${FORCE_DIRECTML},force_ov=${FORCE_OPENVINO},force_cu=${FORCE_CUDA}"

    # Skip GPU setup if environment matches the last successful run
    STAMP_CONTENT=""
    [ -f "$GPU_STAMP" ] && STAMP_CONTENT=$(cat "$GPU_STAMP")

    if [ "$STAMP_CONTENT" = "$GPU_ENV" ]; then
        echo "GPU: Environment unchanged -- skipping GPU setup (delete .gpu-setup-stamp to force)"
    else
        # --- DirectML (WSL2 with GPU passthrough: Qualcomm Adreno, AMD, NVIDIA via D3D12) ---
        # onnxruntime-directml and onnxruntime conflict; swap if needed.
        INSTALL_DIRECTML=0
        if [ "$FORCE_DIRECTML" -eq 1 ]; then
            INSTALL_DIRECTML=1
            echo "GPU: DirectML forced via --force-directml"
        elif [ "$IS_WSL" -eq 1 ] && [ "$HAS_DXG" -eq 1 ]; then
            INSTALL_DIRECTML=1
            echo "GPU: WSL2 with GPU passthrough detected (/dev/dxg) -- installing onnxruntime-directml"
        fi

        if [ "$INSTALL_DIRECTML" -eq 1 ]; then
            if ! python -c "import onnxruntime as ort; assert 'DmlExecutionProvider' in ort.get_available_providers()" &>/dev/null; then
                # onnxruntime-directml conflicts with plain onnxruntime -- try swapping.
                # Note: the pip package is Windows-only; on Linux (including WSL2) it is not
                # published, so we fall back to plain onnxruntime which uses CPU only.
                pip show onnxruntime &>/dev/null && pip uninstall -y onnxruntime || true
                if pip install onnxruntime-directml 2>/dev/null; then
                    echo "GPU: onnxruntime-directml installed (DmlExecutionProvider active)"
                else
                    echo "GPU: onnxruntime-directml not available for this platform -- falling back to plain onnxruntime (CPU only)"
                    pip show onnxruntime &>/dev/null || pip install onnxruntime
                fi
            fi
        fi

        # --- OpenVINO (Intel GPU or x86_64 CPU acceleration; skip on ARM/Qualcomm) ---
        # Not installed when DirectML is already handling acceleration.
        INSTALL_OPENVINO=0
        if [ "$INSTALL_DIRECTML" -eq 0 ]; then
            if [ "$FORCE_OPENVINO" -eq 1 ]; then
                INSTALL_OPENVINO=1
                echo "GPU: OpenVINO forced via --force-openvino"
            elif [ "$HAS_INTEL_GPU" -eq 1 ]; then
                INSTALL_OPENVINO=1
                echo "GPU: Intel GPU detected -- installing OpenVINO for GPU+CPU acceleration"
            elif [ "$ARCH" = "x86_64" ]; then
                INSTALL_OPENVINO=1
                echo "GPU: x86_64 CPU supports OpenVINO CPU acceleration"
            fi
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

        # --- CUDA (future support) ---
        if [ "$FORCE_CUDA" -eq 1 ] || [ "$HAS_NVIDIA_GPU" -eq 1 ]; then
            echo "GPU: NVIDIA GPU detected -- CUDA STT support not yet implemented"
            # Future: pip install faster-whisper[cuda] or similar
        fi

        # Record current environment so next startup can skip this block
        echo "$GPU_ENV" > "$GPU_STAMP"
    fi
fi

# --- systemd user service ---
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_DEST="$SERVICE_DIR/jarvis.service"
SERVICE_SRC="$SCRIPT_DIR/jarvis.service"

mkdir -p "$SERVICE_DIR"

if [ ! -f "$SERVICE_DEST" ] || ! diff -q "$SERVICE_SRC" "$SERVICE_DEST" &>/dev/null; then
    echo "Installing systemd user service..."
    cp "$SERVICE_SRC" "$SERVICE_DEST"
    systemctl --user daemon-reload
fi

if ! systemctl --user is-enabled jarvis &>/dev/null; then
    echo "Enabling jarvis service (start on login)..."
    systemctl --user enable jarvis
fi

echo "Setup OK — all dependencies present."
if [ -z "$INVOCATION_ID" ]; then
    echo "To start now: systemctl --user start jarvis"
    echo "To view logs: journalctl --user -u jarvis -f"
fi
