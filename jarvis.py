#!/usr/bin/env python3
"""Jarvis: Wake-word voice assistant for Claude Code."""

import ctypes
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import json
import logging
import os
from queue import Queue, Empty
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import zoneinfo

# Suppress onnxruntime CUDA warning
import warnings
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
warnings.filterwarnings("ignore", message=".*__array__.*doesn't accept a copy keyword.*")

# Suppress transformers deprecation messages (forced_decoder_ids,
# return_token_timestamps, generation_config defaults, logits processor).
# These come from the transformers custom logger, not the warnings module.
import transformers
transformers.logging.set_verbosity_error()

import numpy as np

# ---------------------------------------------------------------------------
# Platform detection & capability flags
# ---------------------------------------------------------------------------

def _detect_onnx_provider():
    """Detect the best available ONNX Runtime execution provider for this platform.

    Returns provider name string (e.g. "OpenVINOExecutionProvider") or None.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "OpenVINOExecutionProvider" in available:
            return "OpenVINOExecutionProvider"
    except ImportError:
        pass
    return None

_onnx_provider = _detect_onnx_provider()
_OrigOnnxSession = None  # Set by _patch_onnx_providers to bypass monkey-patch

# Capability flags — computed once at import, used to guard optional code paths.
# These check for installed packages, not hardware. Hardware probing is deferred
# to functions like _detect_gpu_devices() which are only called when needed.
_HAS_OPENVINO = _onnx_provider == "OpenVINOExecutionProvider"
_HAS_CUDA = False
try:
    import torch as _torch
    _HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    pass


def _openvino_gpu_available():
    """Check if OpenVINO GPU device is available."""
    if not _HAS_OPENVINO:
        return False
    try:
        from openvino import Core
        return "GPU" in Core().available_devices
    except Exception:
        return False


def _validate_openvino_device(device):
    """Validate that an OpenVINO device exists. Returns the device if valid, or
    falls back to CPU with a warning if the device is not available."""
    if device == "CPU":
        return device
    try:
        from openvino import Core
        available = Core().available_devices
        # Bare "GPU" — resolve to the last (typically discrete/fastest) GPU
        if device == "GPU":
            ov_gpus = [d for b, d in _detect_gpu_devices() if b == "openvino"]
            if ov_gpus:
                resolved = ov_gpus[-1]
                print(f"GPU resolved to {resolved}")
                return resolved
            print(f"WARNING: No OpenVINO GPU devices found. Falling back to CPU.")
            return "CPU"
        # Specific GPU.N — check it exists
        if device in available:
            return device
        print(f"WARNING: OpenVINO device '{device}' not found (available: {', '.join(available)}). Falling back to CPU.")
        return "CPU"
    except Exception:
        print(f"WARNING: Could not query OpenVINO devices. Falling back to CPU.")
        return "CPU"


def _detect_gpu_devices():
    """Detect available GPU devices for STT acceleration.

    Returns a list of (backend, device) tuples, e.g.
    [("openvino", "GPU.0"), ("openvino", "GPU.1"), ("cuda", "cuda:0")].
    CPU devices are not included (always assumed available).
    """
    devices = []

    # OpenVINO GPUs
    if _HAS_OPENVINO:
        try:
            import onnxruntime as ort
            import openwakeword
            probe_path = os.path.join(
                os.path.dirname(openwakeword.__file__),
                "resources", "models", "melspectrogram.onnx",
            )
            for i in range(8):
                dev = f"GPU.{i}"
                try:
                    fd = os.dup(2)
                    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
                    try:
                        sess = ort.InferenceSession(probe_path, providers=[
                            ("OpenVINOExecutionProvider", {"device_type": dev}),
                            "CPUExecutionProvider",
                        ])
                        active = sess.get_providers()
                    finally:
                        os.dup2(fd, 2)
                        os.close(fd)
                    if "OpenVINOExecutionProvider" not in active:
                        break
                    devices.append(("openvino", dev))
                except Exception:
                    break
        except ImportError:
            pass

    # CUDA GPUs (future)
    if _HAS_CUDA:
        import torch
        for i in range(torch.cuda.device_count()):
            devices.append(("cuda", f"cuda:{i}"))

    return devices


def _generate_bench_audio(duration_s=2.0, sr=16000):
    """Generate a short sine wave WAV file for benchmarking. Returns path."""
    audio = (0.3 * np.sin(2 * np.pi * 200 *
             np.linspace(0, duration_s, int(sr * duration_s),
                         dtype=np.float32)) * 32767).astype(np.int16)
    path = os.path.join(tempfile.gettempdir(), "stt_bench.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return path


def _ensure_openvino_cache(model_name):
    """Export Whisper to OpenVINO IR if not already cached. Returns cache dir."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(_script_dir, ".cache", f"whisper-{model_name}-openvino")
    if os.path.exists(os.path.join(cache_dir, "openvino_encoder_model.xml")):
        return cache_dir

    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    hf_model = f"openai/whisper-{model_name}"
    print(f"  Exporting whisper-{model_name} to OpenVINO (one-time, ~45s)...")
    fd = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            hf_model, export=True, device="CPU", compile=False)
    finally:
        os.dup2(fd, 2)
        os.close(fd)
    os.makedirs(cache_dir, exist_ok=True)
    model.save_pretrained(cache_dir)
    AutoProcessor.from_pretrained(hf_model).save_pretrained(cache_dir)
    del model
    print("  Export complete.")
    return cache_dir


def _bench_openvino_device(cache_dir, device, audio_path):
    """Benchmark a single OpenVINO device. Returns mean seconds or None on failure."""
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline
    try:
        fd = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        try:
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                cache_dir, device=device, compile=True)
            processor = AutoProcessor.from_pretrained(cache_dir)
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        pipe(audio_path)  # warmup
        t0 = time.perf_counter()
        pipe(audio_path)
        pipe(audio_path)
        elapsed = (time.perf_counter() - t0) / 2
        del model, pipe
        return elapsed
    except Exception as e:
        print(f"  openvino {device}: failed ({e})")
        return None


def _bench_faster_whisper(model_name, audio_path):
    """Benchmark faster-whisper on CPU. Returns mean seconds or None on failure."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        # warmup
        list(model.transcribe(audio_path, beam_size=1)[0])
        t0 = time.perf_counter()
        list(model.transcribe(audio_path, beam_size=1)[0])
        list(model.transcribe(audio_path, beam_size=1)[0])
        elapsed = (time.perf_counter() - t0) / 2
        del model
        return elapsed
    except Exception as e:
        print(f"  faster-whisper: failed ({e})")
        return None


def _resolve_stt_auto(model_name):
    """Resolve the best STT backend and device, with caching.

    Benchmarks faster-whisper CPU against all available accelerated backends.
    Caches the result in .cache/stt_auto.json. Re-benchmarks if hardware or model changes.

    Returns (backend, device) tuple, e.g. ("openvino", "GPU.1") or ("faster-whisper", None).
    """
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(_script_dir, ".cache", "stt_auto.json")

    gpu_devices = _detect_gpu_devices()
    current_hw = sorted(f"{b}:{d}" for b, d in gpu_devices)

    # Check cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if (cached.get("hardware") == current_hw
                    and cached.get("model") == model_name
                    and cached.get("has_openvino") == _HAS_OPENVINO):
                backend = cached["backend"]
                device = cached.get("device")
                print(f"STT auto (cached): {backend}"
                      + (f" on {device}" if device else ""))
                return backend, device
        except Exception:
            pass

    # Benchmark
    print("STT auto-detect: benchmarking backends...")
    audio_path = _generate_bench_audio()
    results = {}

    # faster-whisper CPU (always available)
    print("  faster-whisper CPU...", end=" ", flush=True)
    fw_time = _bench_faster_whisper(model_name, audio_path)
    if fw_time is not None:
        results[("faster-whisper", None)] = fw_time
        print(f"{fw_time*1000:.0f} ms")

    # OpenVINO: CPU + any GPU devices
    if _HAS_OPENVINO:
        try:
            cache_dir = _ensure_openvino_cache(model_name)
            ov_gpu_devs = [d for b, d in gpu_devices if b == "openvino"]
            for dev in ["CPU"] + ov_gpu_devs:
                print(f"  openvino {dev}...", end=" ", flush=True)
                ov_time = _bench_openvino_device(cache_dir, dev, audio_path)
                if ov_time is not None:
                    results[("openvino", dev)] = ov_time
                    print(f"{ov_time*1000:.0f} ms")
        except Exception as e:
            print(f"  openvino setup failed: {e}")

    # CUDA (future): would add _bench_cuda_whisper here

    os.unlink(audio_path)

    if not results:
        print("STT auto-detect: no backends available, defaulting to faster-whisper")
        return "faster-whisper", None

    # Pick winner
    (best_backend, best_device), best_time = min(results.items(), key=lambda x: x[1])

    # Cache result
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "backend": best_backend,
            "device": best_device,
            "hardware": current_hw,
            "has_openvino": _HAS_OPENVINO,
            "model": model_name,
            "time_ms": round(best_time * 1000),
        }, f, indent=2)
    label = best_backend + (f" on {best_device}" if best_device else "")
    print(f"STT auto-detect selected: {label} ({best_time*1000:.0f} ms)")
    return best_backend, best_device


def _resolve_stt_openvino_device(model_name):
    """Resolve the best OpenVINO device for STT, with caching.

    Benchmarks all available OpenVINO devices, caches the winner in
    .cache/stt_auto_device.json. Re-benchmarks if the device list changes.

    Returns the device string (e.g. "GPU.1", "CPU").
    """
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(_script_dir, ".cache", "stt_auto_device.json")

    ov_gpu_devs = [d for b, d in _detect_gpu_devices() if b == "openvino"]
    current_hw = sorted(ov_gpu_devs)

    # Check cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if (cached.get("gpu_devices") == current_hw
                    and cached.get("model") == model_name):
                device = cached["device"]
                print(f"STT auto-device (cached): {device}")
                return device
        except Exception:
            pass

    # Benchmark all candidates
    candidates = ["CPU"] + ov_gpu_devs
    print(f"STT auto-detect: benchmarking {', '.join(candidates)}...")

    audio_path = _generate_bench_audio()
    cache_dir = _ensure_openvino_cache(model_name)

    best_device = "CPU"
    best_time = float("inf")

    for dev in candidates:
        ov_time = _bench_openvino_device(cache_dir, dev, audio_path)
        if ov_time is not None:
            print(f"  {dev}: {ov_time*1000:.0f} ms")
            if ov_time < best_time:
                best_time = ov_time
                best_device = dev

    os.unlink(audio_path)

    # Cache result
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({
            "device": best_device,
            "gpu_devices": current_hw,
            "model": model_name,
            "time_ms": round(best_time * 1000),
        }, f, indent=2)
    print(f"STT auto-device selected: {best_device} ({best_time*1000:.0f} ms)")
    return best_device


def _patch_onnx_providers(device_type="CPU"):
    """Monkey-patch onnxruntime.InferenceSession to inject OpenVINO provider.

    Libraries like openwakeword, silero-vad, piper, and kokoro hardcode
    CPUExecutionProvider. This patch transparently upgrades them to use
    OpenVINO when available. device_type selects CPU, GPU, or AUTO
    acceleration. Models incompatible with the provider silently fall back
    to CPU.
    """
    if _onnx_provider is None:
        return

    import onnxruntime as ort
    global _OrigOnnxSession
    _OrigSession = ort.InferenceSession
    _OrigOnnxSession = _OrigSession

    ov_provider = (_onnx_provider, {"device_type": device_type})

    class _PatchedSession(_OrigSession):
        def __init__(self, *args, providers=None, **kwargs):
            orig_providers = providers
            if providers is not None:
                patched = []
                for p in providers:
                    name = p if isinstance(p, str) else p[0]
                    if name == "CPUExecutionProvider":
                        patched.append(ov_provider)
                        patched.append("CPUExecutionProvider")
                    else:
                        patched.append(p)
                providers = patched
            else:
                providers = [ov_provider, "CPUExecutionProvider"]
            try:
                # Suppress C++ error messages from onnxruntime during probe
                _fd = os.dup(2)
                os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
                try:
                    super().__init__(*args, providers=providers, **kwargs)
                finally:
                    os.dup2(_fd, 2)
                    os.close(_fd)
            except Exception:
                # Model incompatible with OpenVINO — fall back to original providers
                fallback = orig_providers or ["CPUExecutionProvider"]
                super().__init__(*args, providers=fallback, **kwargs)

    ort.InferenceSession = _PatchedSession

# ---------------------------------------------------------------------------
# Load configuration from config/jarvis.ini
# ---------------------------------------------------------------------------
import configparser

_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
_cfg = configparser.ConfigParser()
if not _cfg.read(os.path.join(_config_dir, "jarvis.ini")):
    raise FileNotFoundError(f"Config file not found: {os.path.join(_config_dir, 'jarvis.ini')}")
_cfg.read(os.path.join(_config_dir, "secrets.ini"))

def _parse_message_list(value):
    return [m.strip() for m in value.split(",") if m.strip()]

DEBUG_MODELS = _cfg.getboolean("debug", "models", fallback=False)
DEBUG_RECORDING = _cfg.getboolean("debug", "recording", fallback=False)
DEBUG_TRANSCRIPTION = _cfg.getboolean("debug", "transcription", fallback=False)
DEBUG_CLAUDE = _cfg.getboolean("debug", "claude", fallback=False)
DEBUG_TTS = _cfg.getboolean("debug", "tts", fallback=False)
DEBUG_ECHO = _cfg.getboolean("debug", "echo", fallback=False)

# ---------------------------------------------------------------------------
# Debug / profiling helpers
# ---------------------------------------------------------------------------
_t0 = time.perf_counter()
_debug_logger = None

def _get_debug_logger():
    """Lazily initialize the debug file logger (needs SCRIPT_DIR which is set later)."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = logging.getLogger("jarvis.debug")
        _debug_logger.setLevel(logging.DEBUG)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _debug_logs_dir = os.path.join(_script_dir, "logs")
        os.makedirs(_debug_logs_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(_debug_logs_dir, "debug.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _debug_logger.addHandler(handler)
    return _debug_logger

def debug_log(flag, msg):
    """Log a timestamped debug message to logs/debug.log when the given flag is enabled."""
    if flag:
        elapsed = time.perf_counter() - _t0
        _get_debug_logger().debug("[+%8.3fs] %s", elapsed, msg)

@contextmanager
def debug_timer(flag, label):
    """Context manager that logs elapsed time for a block when flag is enabled."""
    if flag:
        start = time.perf_counter()
        debug_log(flag, f"{label} — started")
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            debug_log(flag, f"{label} — finished in {dt:.3f}s")
    else:
        yield

RATE = _cfg.getint("audio", "rate")
CHANNELS = _cfg.getint("audio", "channels")
CHUNK = _cfg.getint("audio", "chunk")
VAD_THRESHOLD = _cfg.getfloat("vad", "threshold")
VAD_CHUNK = _cfg.getint("vad", "chunk")
SILENCE_DURATION = _cfg.getfloat("vad", "silence_duration")
PRE_SPEECH_TIMEOUT = _cfg.getfloat("vad", "pre_speech_timeout")
SILENCE_RATIO = _cfg.getfloat("vad", "silence_ratio")
MAX_RECORD_SECONDS = _cfg.getint("vad", "max_record_seconds")
WAKE_WORD_NAME = _cfg.get("wake_word", "name")
WAKE_WORD_THRESHOLD = _cfg.getfloat("wake_word", "threshold")
WAKE_WORD_MODEL_PATH = _cfg.get("wake_word", "model_path", fallback="")
STT_MODEL = _cfg.get("stt", "model")
STT_BACKEND = _cfg.get("stt", "backend", fallback="faster-whisper").lower()
STT_OPENVINO_DEVICE = _cfg.get("stt", "openvino_device", fallback="AUTO").upper()
CLAUDE_TIMEOUT = _cfg.getint("claude", "timeout")
TTS_ENGINE = _cfg.get("tts", "engine")
TTS_VOICE = _cfg.get("tts", "voice")
TTS_OPENVINO_DEVICE = _cfg.get("tts", "openvino_device", fallback="CPU").upper()

# Apply ONNX provider patch with CPU for general models (wake word, VAD, STT).
# GPU/AUTO patch is applied temporarily during TTS loading only, since OpenVINO
# GPU produces subtly incorrect numerical results in openwakeword's melspectrogram
# and embedding models, breaking wake word detection.
if TTS_OPENVINO_DEVICE == "CPU":
    _patch_onnx_providers(device_type="CPU")

COMMERCIAL_USE = _cfg.getboolean("licensing", "commercial_use", fallback=False)

TELEGRAM_TOKEN = _cfg.get("telegram", "bot_token", fallback="")
_telegram_allowed_raw = _cfg.get("telegram", "allowed_users", fallback="")
TELEGRAM_ALLOWED_USER_IDS = (
    {int(uid.strip()) for uid in _telegram_allowed_raw.split(",") if uid.strip()}
    if _telegram_allowed_raw else set()
)
TELEGRAM_VOICE_REPLIES = _cfg.getboolean("telegram", "voice_replies", fallback=True)

INITIAL_ACK_DELAY = _cfg.getfloat("timing", "initial_ack_delay")
STILL_WORKING_INTERVAL = _cfg.getint("timing", "still_working_interval")
ACKNOWLEDGEMENTS = _parse_message_list(_cfg.get("messages", "acknowledgements"))
STILL_WORKING = _parse_message_list(_cfg.get("messages", "still_working"))

def _apply_wake_word_name(messages):
    """Replace 'Jarvis' in config messages with the configured wake word name."""
    wn = WAKE_WORD_NAME.capitalize()
    return [re.sub(r'\bJarvis\b', wn, m) for m in messages]

STARTUP_MESSAGES = _apply_wake_word_name(_parse_message_list(_cfg.get("messages", "startup")))
SHUTDOWN_MESSAGES = _apply_wake_word_name(_parse_message_list(_cfg.get("messages", "shutdown")))

# Piper voices known to have non-commercial licenses (CC BY-NC-SA 4.0 or research-only)
_NC_PIPER_VOICES = {
    "en_US-lessac-low", "en_US-lessac-medium", "en_US-lessac-high",
    "en_US-amy-low", "en_US-amy-medium",
    "en_US-ryan-low", "en_US-ryan-medium", "en_US-ryan-high",
}


def _validate_commercial_use():
    """Check that all components have commercially-compatible licenses."""
    errors = []

    # Wake word: bundled openwakeword models are CC BY-NC-SA 4.0
    if not WAKE_WORD_MODEL_PATH:
        errors.append(
            "Wake word: bundled openwakeword models are CC BY-NC-SA 4.0 (non-commercial).\n"
            "  Set [wake_word] model_path to a custom-trained model for commercial use."
        )

    # TTS voice license check
    if TTS_ENGINE == "piper" and TTS_VOICE in _NC_PIPER_VOICES:
        errors.append(
            f"TTS voice '{TTS_VOICE}' has a non-commercial license.\n"
            f"  Use a [commercial] voice (see jarvis.ini) or switch to kokoro engine."
        )

    if errors:
        print("\n=== COMMERCIAL USE LICENSE CHECK FAILED ===")
        for err in errors:
            print(f"  * {err}")
        print("Set [licensing] commercial_use = false to disable this check.\n")
        sys.exit(1)


if COMMERCIAL_USE:
    _validate_commercial_use()

# PulseAudio simple API via ctypes
try:
    _pulse_simple = ctypes.cdll.LoadLibrary("libpulse-simple.so.0")
except OSError:
    print("ERROR: libpulse-simple.so.0 not found. Install PulseAudio: sudo apt-get install libpulse0")
    sys.exit(1)

# pa_sample_format_t: PA_SAMPLE_S16LE = 3
# pa_stream_direction_t: PA_STREAM_RECORD = 2
class _PaSampleSpec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]

_pulse_simple.pa_simple_new.restype = ctypes.c_void_p
_pulse_simple.pa_simple_new.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(_PaSampleSpec), ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]
_pulse_simple.pa_simple_read.restype = ctypes.c_int
_pulse_simple.pa_simple_read.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
]
_pulse_simple.pa_simple_free.restype = None
_pulse_simple.pa_simple_free.argtypes = [ctypes.c_void_p]
_pulse_simple.pa_simple_write.restype = ctypes.c_int
_pulse_simple.pa_simple_write.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
]
_pulse_simple.pa_simple_drain.restype = ctypes.c_int
_pulse_simple.pa_simple_drain.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
]
_pulse_simple.pa_simple_flush.restype = ctypes.c_int
_pulse_simple.pa_simple_flush.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
]


class PulseRecorder:
    """Record audio via PulseAudio simple API."""

    def __init__(self, rate, channels, chunk):
        self.chunk = chunk
        self.frame_size = 2 * channels  # 16-bit samples
        spec = _PaSampleSpec(format=3, rate=rate, channels=channels)
        err = ctypes.c_int(0)
        self._pa = _pulse_simple.pa_simple_new(
            None, b"jarvis", 2, None, b"record",
            ctypes.byref(spec), None, None, ctypes.byref(err),
        )
        if not self._pa:
            raise RuntimeError(f"pa_simple_new failed: error {err.value}")

    def read(self, n_frames, exception_on_overflow=False):
        n_bytes = n_frames * self.frame_size
        buf = ctypes.create_string_buffer(n_bytes)
        err = ctypes.c_int(0)
        ret = _pulse_simple.pa_simple_read(self._pa, buf, n_bytes, ctypes.byref(err))
        if ret < 0:
            raise RuntimeError(f"pa_simple_read failed: error {err.value}")
        return buf.raw

    def flush(self):
        """Read and discard buffered audio to clear stale data."""
        err = ctypes.c_int(0)
        _pulse_simple.pa_simple_flush(self._pa, ctypes.byref(err))

    def close(self):
        if self._pa:
            _pulse_simple.pa_simple_free(self._pa)
            self._pa = None

class PulsePlayer:
    """Play audio via PulseAudio simple API."""

    def __init__(self, rate, channels=1):
        self.frame_size = 2 * channels  # 16-bit samples
        spec = _PaSampleSpec(format=3, rate=rate, channels=channels)
        err = ctypes.c_int(0)
        self._pa = _pulse_simple.pa_simple_new(
            None, b"jarvis", 1, None, b"playback",
            ctypes.byref(spec), None, None, ctypes.byref(err),
        )
        if not self._pa:
            raise RuntimeError(f"pa_simple_new (playback) failed: error {err.value}")

    def write(self, data):
        err = ctypes.c_int(0)
        ret = _pulse_simple.pa_simple_write(self._pa, data, len(data), ctypes.byref(err))
        if ret < 0:
            raise RuntimeError(f"pa_simple_write failed: error {err.value}")

    def drain(self):
        err = ctypes.c_int(0)
        _pulse_simple.pa_simple_drain(self._pa, ctypes.byref(err))

    def flush(self):
        err = ctypes.c_int(0)
        _pulse_simple.pa_simple_flush(self._pa, ctypes.byref(err))

    def close(self):
        if self._pa:
            _pulse_simple.pa_simple_free(self._pa)
            self._pa = None


def reset_wake_model(wake_model):
    """Fully reset wake word model, including preprocessor feature buffers."""
    wake_model.reset()
    # reset() only clears the prediction buffer. The preprocessor still holds
    # old audio features (raw_data_buffer, melspectrogram_buffer, feature_buffer)
    # which can cause false re-triggers from the sliding window.
    pp = wake_model.preprocessor
    pp.raw_data_buffer.clear()
    pp.melspectrogram_buffer = np.ones((76, 32))
    pp.accumulated_samples = 0
    pp.feature_buffer = pp._get_embeddings(np.zeros(16000 * 10).astype(np.int16))


def _load_openvino_whisper(model_name, device=None):
    """Load Whisper via OpenVINO, exporting and caching the IR model on first run.

    Returns an object with a `transcribe(audio_path)` method that returns text.
    """
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline

    # Resolve device: AUTO benchmarks all devices (CPU+GPU) to find the best
    if device is None or device == "AUTO":
        device = _resolve_stt_openvino_device(model_name)
    else:
        device = _validate_openvino_device(device)

    cache_dir = _ensure_openvino_cache(model_name)

    # Load from cache
    print(f"  Loading from cache ({device})...")
    fd = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            cache_dir, device=device, compile=True)
        processor = AutoProcessor.from_pretrained(cache_dir)
    finally:
        os.dup2(fd, 2)
        os.close(fd)

    # Suppress misleading "Device set to use cpu" warning from transformers.
    # The pipeline reports PyTorch device=cpu, but OpenVINO models use their
    # own device assignment and ignore PyTorch's — the model runs on `device`.
    import logging as _logging
    _tf_logger = _logging.getLogger("transformers.pipelines.base")
    _prev_level = _tf_logger.level
    _tf_logger.setLevel(_logging.ERROR)
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
    finally:
        _tf_logger.setLevel(_prev_level)

    class _OVWhisperModel:
        """Wrapper providing unified transcribe(audio_path) -> str interface."""
        backend = "openvino"

        def __init__(self, pipe, device):
            self._pipe = pipe
            self.device = device

        def transcribe(self, audio_path):
            return self._pipe(audio_path)["text"].strip()

    return _OVWhisperModel(pipe, device)


def load_models():
    """Load wake word and whisper models."""
    with debug_timer(DEBUG_MODELS, "load_models total"):
        if _onnx_provider:
            print(f"ONNX provider: {_onnx_provider} ({TTS_OPENVINO_DEVICE})")
        else:
            print("ONNX provider: CPUExecutionProvider")
        print("Loading wake word model...")
        with debug_timer(DEBUG_MODELS, "load wake word model"):
            from openwakeword.model import Model as WakeModel
            if WAKE_WORD_MODEL_PATH:
                _ww_path = WAKE_WORD_MODEL_PATH
            else:
                _ww_path = os.path.join(
                    os.path.dirname(__import__('openwakeword').__file__),
                    "resources", "models", f"hey_{WAKE_WORD_NAME}_v0.1.onnx")
            wake_model = WakeModel(wakeword_model_paths=[_ww_path])

        # Resolve backend (auto picks the fastest via benchmark)
        stt_backend = STT_BACKEND
        stt_ov_device = STT_OPENVINO_DEVICE
        if stt_backend == "auto":
            stt_backend, stt_ov_device = _resolve_stt_auto(STT_MODEL)
            if stt_ov_device is None:
                stt_ov_device = "CPU"

        print(f"Loading whisper model ({STT_MODEL}, {stt_backend}"
              + (f" on {stt_ov_device}" if stt_backend == "openvino" else "") + ")...")
        with debug_timer(DEBUG_MODELS, "load whisper model"):
            if stt_backend == "openvino":
                whisper_model = _load_openvino_whisper(STT_MODEL, stt_ov_device)
            else:
                from faster_whisper import WhisperModel as _FWModel
                _fw = _FWModel(STT_MODEL, device="cpu", compute_type="int8")

                class _FasterWhisperModel:
                    """Wrapper providing unified transcribe(audio_path) -> str interface."""
                    backend = "faster-whisper"
                    device = "cpu"

                    def __init__(self, model):
                        self._model = model

                    def transcribe(self, audio_path):
                        segments, _ = self._model.transcribe(
                            audio_path, beam_size=1, without_timestamps=True,
                            vad_filter=True)
                        return " ".join(seg.text.strip() for seg in segments).strip()

                whisper_model = _FasterWhisperModel(_fw)

        print("Loading Silero VAD model...")
        with debug_timer(DEBUG_MODELS, "load Silero VAD model"):
            from silero_vad import load_silero_vad
            vad_model = load_silero_vad(onnx=True)

        print(f"Loading TTS model ({TTS_ENGINE}: {TTS_VOICE})...")
        with debug_timer(DEBUG_MODELS, "load TTS model"):
            # Temporarily switch to GPU provider for TTS loading if configured.
            tts_device = TTS_OPENVINO_DEVICE
            if tts_device != "CPU" and _onnx_provider is not None:
                tts_device = _validate_openvino_device(tts_device)
                if tts_device != "CPU":
                    _patch_onnx_providers(device_type=tts_device)

            if TTS_ENGINE == "piper":
                from piper import PiperVoice
                # Download voice if needed
                voice_dir = os.path.join(os.path.dirname(__file__), "voices")
                voice_path = os.path.join(voice_dir, f"{TTS_VOICE}.onnx")
                voice_config = voice_path + ".json"

                if not os.path.exists(voice_path):
                    os.makedirs(voice_dir, exist_ok=True)
                    print(f"Downloading TTS voice ({TTS_VOICE})...")
                    print(f"NOTE: Piper voice '{TTS_VOICE}' may be under a non-commercial license")
                    print(f"      (e.g. CC BY-NC-SA 4.0). Check the model card before redistributing.")
                    print(f"      See https://huggingface.co/rhasspy/piper-voices")
                    import urllib.request
                    import urllib.error
                    # Parse voice name: en_US-lessac-medium -> en/en_US/lessac/medium
                    parts = TTS_VOICE.split("-")
                    lang = parts[0]                    # en_US
                    lang_family = lang.split("_")[0]   # en
                    dataset = parts[1]                 # lessac
                    quality = parts[2]                 # medium
                    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang}/{dataset}/{quality}"
                    try:
                        urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx", voice_path)
                        urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx.json", voice_config)
                    except (urllib.error.URLError, OSError) as e:
                        # Clean up partial downloads
                        for f in (voice_path, voice_config):
                            if os.path.exists(f):
                                os.unlink(f)
                        raise RuntimeError(f"Failed to download TTS voice '{TTS_VOICE}': {e}") from e

                tts_voice = PiperVoice.load(voice_path)
                if tts_device != "CPU" and _onnx_provider is not None:
                    # Warmup: first GPU inference compiles the graph (~16s)
                    print("Warming up GPU TTS (first inference)...")
                    for _ in tts_voice.synthesize("warmup"):
                        pass
                    print("GPU TTS ready.")
            elif TTS_ENGINE == "kokoro":
                os.environ["HF_HUB_OFFLINE"] = "1"
                from kokoro_onnx import Kokoro
                voice_dir = os.path.join(os.path.dirname(__file__), "voices")
                model_path = os.path.join(voice_dir, "kokoro-v1.0.onnx")
                voices_path = os.path.join(voice_dir, "voices-v1.0.bin")
                if not os.path.exists(model_path) or not os.path.exists(voices_path):
                    raise FileNotFoundError(
                        f"Kokoro model files not found in {voice_dir}. "
                        "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                        "https://github.com/thewh1teagle/kokoro-onnx/releases"
                    )
                tts_voice = KokoroTTS(Kokoro(model_path, voices_path), TTS_VOICE)
            else:
                raise ValueError(f"Unsupported TTS engine: {TTS_ENGINE}")

            # Restore original ONNX session class so future sessions aren't patched
            if tts_device != "CPU" and _OrigOnnxSession is not None:
                import onnxruntime as ort
                ort.InferenceSession = _OrigOnnxSession

    print("All models loaded.")
    return wake_model, whisper_model, tts_voice, vad_model


def record_until_silence(stream, vad_model, pre_roll=None):
    """Record audio until speech is followed by silence using Silero VAD. Returns raw audio bytes.

    pre_roll: optional list of raw audio byte chunks to prepend (captures speech
    that arrived between wake word detection and the start of recording).
    """
    import torch

    print("Listening...")
    debug_log(DEBUG_RECORDING, "record_until_silence — started")
    rec_start = time.perf_counter()
    vad_model.reset_states()
    frames = list(pre_roll) if pre_roll else []
    if pre_roll:
        debug_log(DEBUG_RECORDING, f"  pre-roll: {len(list(pre_roll))} chunks")
    speech_detected = False
    chunks_per_second = RATE / VAD_CHUNK
    silence_window_size = int(SILENCE_DURATION * chunks_per_second)
    pre_speech_chunks = int(PRE_SPEECH_TIMEOUT * chunks_per_second)

    # Rolling window: track whether each recent chunk was silent
    window = deque(maxlen=silence_window_size)

    for chunk_idx in range(int(MAX_RECORD_SECONDS * chunks_per_second)):
        try:
            data = stream.read(VAD_CHUNK, exception_on_overflow=False)
        except RuntimeError:
            print("ERROR: Audio device disconnected during recording.")
            break
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(audio_data)
        speech_prob = vad_model(chunk_tensor, RATE).item()

        is_silent = speech_prob < VAD_THRESHOLD

        if not is_silent and not speech_detected:
            speech_detected = True
            debug_log(DEBUG_RECORDING, f"  speech onset at chunk {chunk_idx} ({chunk_idx / chunks_per_second:.2f}s)")

        # Give up if no speech detected within the pre-speech timeout
        if not speech_detected and chunk_idx >= pre_speech_chunks:
            print("No speech detected, giving up.")
            debug_log(DEBUG_RECORDING, f"  no speech after {PRE_SPEECH_TIMEOUT}s, giving up")
            return b""

        if speech_detected:
            window.append(is_silent)

        # Stop when the rolling window is full and mostly silent
        if speech_detected and len(window) == silence_window_size:
            if sum(window) / silence_window_size >= SILENCE_RATIO:
                debug_log(DEBUG_RECORDING, f"  silence detected at chunk {chunk_idx} ({chunk_idx / chunks_per_second:.2f}s)")
                break

    rec_dt = time.perf_counter() - rec_start
    n_frames = len(frames)
    audio_duration = n_frames * VAD_CHUNK / RATE
    debug_log(DEBUG_RECORDING, f"record_until_silence — finished in {rec_dt:.3f}s (captured {audio_duration:.2f}s of audio, {n_frames} chunks)")
    print("Done recording.")
    return b"".join(frames)


def transcribe(whisper_model, audio_bytes):
    """Transcribe raw audio bytes with whisper (faster-whisper or OpenVINO)."""
    debug_log(DEBUG_TRANSCRIPTION, "transcribe — started")
    t_start = time.perf_counter()

    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)
    t_wav = time.perf_counter()
    debug_log(DEBUG_TRANSCRIPTION, f"  WAV write: {t_wav - t_start:.3f}s ({len(audio_bytes)} bytes)")

    try:
        text = whisper_model.transcribe(tmp_path)
        t_done = time.perf_counter()
        backend = getattr(whisper_model, 'backend', 'unknown')
        debug_log(DEBUG_TRANSCRIPTION, f"  {backend} transcribe: {t_done - t_wav:.3f}s")
        debug_log(DEBUG_TRANSCRIPTION, f"transcribe — finished in {t_done - t_start:.3f}s, result: '{text[:80]}'")
        return text
    finally:
        os.unlink(tmp_path)


# Known Whisper hallucination strings — phantom text produced on silence/noise.
# These are well-documented artifacts from Whisper's training data (YouTube).
WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "thank you",
    "thanks for listening",
    "thank you for listening",
    "please subscribe",
    "like and subscribe",
    "subscribe",
    "subscribe to my channel",
    "see you next time",
    "see you in the next video",
    "see you in the next one",
    "the end",
}


def is_garbage_transcription(text):
    """Filter out Whisper hallucinations and punctuation-only output."""
    # Strip all non-alphanumeric characters — reject if nothing remains
    if not re.sub(r'[^a-zA-Z0-9]', '', text):
        return True
    # Check against known Whisper hallucination patterns
    return _normalize(text) in WHISPER_HALLUCINATIONS


def clean_text_for_speech(text, keep_wake_word=False):
    """Strip markdown formatting that TTS would speak literally."""
    import re
    # Remove fenced code blocks (```...```) — replace with brief mention
    text = re.sub(r'```[^\n]*\n.*?```', ' (code omitted) ', text, flags=re.DOTALL)
    # Remove inline code WITH content (`...`) — replace with brief mention
    text = re.sub(r'`[^`]+`', ' (code) ', text)
    # Remove bold/italic markers (**, *, __, _)
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'_{1,3}', '', text)
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove any remaining backticks
    text = re.sub(r'`', '', text)
    # Remove bullet point markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Replace wake word so TTS doesn't trigger the wake word detector
    if not keep_wake_word:
        text = re.sub(r'\b' + re.escape(WAKE_WORD_NAME) + r'\b', 'wake word', text, flags=re.IGNORECASE)
    # Collapse excess whitespace from removals
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Track recently spoken text for echo detection — list of (timestamp, text) tuples.
# Time-based window instead of fixed count so rapid interactions don't evict entries.
_recently_spoken = []
ECHO_MEMORY_SECONDS = 60
ECHO_SIMILARITY_THRESHOLD = 0.7
# Canned messages (acks, fillers) are short common phrases prone to matching
# legitimate user speech at the standard threshold, so use a stricter one.
ECHO_CANNED_THRESHOLD = 0.85
# If the same echo-like text is heard this many times consecutively,
# treat it as intentional speech rather than an echo.
ECHO_REPEAT_THRESHOLD = 2
_last_echo_text = None
_echo_repeat_count = 0

# Built-in command responses (not from config, but need echo filtering)
BUILTIN_RESPONSES = [
    "Restarting now.",
    "Restarting now",
    "Reverted commit. Restarting now.",
    "Sorry, I couldn't find the git repository.",
    "Sorry, the revert failed.",
    "The queue is empty.",
    "Queue cleared.",
    "All queued prompts have been processed.",
    "Stopped.",
    "Queued. I'll handle that after this task.",
    "Shutting down. Goodbye.",
]

# Pre-normalized set of all canned messages for fast echo lookup
_CANNED_MESSAGES = (
    ACKNOWLEDGEMENTS + STILL_WORKING + STARTUP_MESSAGES + SHUTDOWN_MESSAGES
    + BUILTIN_RESPONSES
)
_canned_normalized = None  # lazily built


def _normalize(text):
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _get_canned_normalized():
    """Return set of normalized canned messages (built once).

    Includes both original and TTS-cleaned versions so echoes of spoken
    text are caught (e.g. TTS replaces 'Jarvis' with 'wake word').
    """
    global _canned_normalized
    if _canned_normalized is None:
        _canned_normalized = set()
        for m in _CANNED_MESSAGES:
            _canned_normalized.add(_normalize(m))
            _canned_normalized.add(_normalize(clean_text_for_speech(m)))
    return _canned_normalized


def _matches_any(norm_t, candidates, threshold=ECHO_SIMILARITY_THRESHOLD):
    """Check if normalized text matches any candidate via substring or fuzzy match."""
    for norm_s in candidates:
        if not norm_s:
            continue
        # Substring match: only when user text is a shorter/equal version of spoken
        # text (norm_t in norm_s). We intentionally do NOT check the reverse
        # (norm_s in norm_t) because that catches legitimate user speech that
        # happens to contain a canned phrase (e.g. "Give me a second opinion"
        # would match the canned "Give me a second").
        # Min ratio 0.4 allows partial echo fragments of long sentences.
        len_ratio = len(norm_t) / len(norm_s) if norm_s else 0
        if 0.4 <= len_ratio <= 2.0 and norm_t in norm_s:
            return True
        ratio = SequenceMatcher(None, norm_t, norm_s).ratio()
        if ratio >= threshold:
            return True
    return False


def is_self_echo(transcribed):
    """Check if transcribed text is an echo of something Jarvis recently said."""
    norm_t = _normalize(transcribed)
    if not norm_t:
        return False
    # Check against canned messages with stricter threshold — canned phrases
    # are short and common, so the standard threshold causes false positives
    # (e.g. "Give me a second opinion" matching "Give me a second").
    if _matches_any(norm_t, _get_canned_normalized(), threshold=ECHO_CANNED_THRESHOLD):
        return True
    # Check against recently spoken dynamic text (time-windowed).
    # Also split long responses into sentences so partial echoes
    # (e.g. last sentence of a multi-sentence response) are caught.
    now = time.time()
    _recently_spoken[:] = [(ts, t) for ts, t in _recently_spoken
                           if now - ts <= ECHO_MEMORY_SECONDS]
    spoken_norms = []
    for _ts, text in _recently_spoken:
        spoken_norms.append(_normalize(text))
        # Split into sentences for fragment matching on long responses
        for sentence in re.split(r'[.!?]+', text):
            sentence = sentence.strip()
            if len(sentence) > 3:  # skip trivially short fragments
                spoken_norms.append(_normalize(sentence))
    if _matches_any(norm_t, spoken_norms):
        return True
    return False


# ---------------------------------------------------------------------------
# Persistent prompt queue — survives restarts, auto-retries after rate limits
# ---------------------------------------------------------------------------
QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prompt_queue.json")


def load_queue():
    """Load the prompt queue from disk. Returns [] on missing/corrupt file."""
    try:
        with open(QUEUE_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


def save_queue(queue):
    """Atomically write the queue to disk."""
    try:
        os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(queue, f)
        os.replace(tmp, QUEUE_FILE)
    except OSError as e:
        print(f"WARNING: Could not save queue: {e}")


def queue_add(prompt, position=0):
    """Insert a prompt into the queue (front by default). Returns updated queue."""
    q = load_queue()
    entry = {"prompt": prompt, "queued_at": datetime.now().isoformat()}
    q.insert(position, entry)
    save_queue(q)
    return q


def queue_pop():
    """Remove and return the first queue entry, or None."""
    q = load_queue()
    if not q:
        return None
    entry = q.pop(0)
    save_queue(q)
    return entry


def queue_list():
    """Return the queue without modifying it."""
    return load_queue()


def queue_clear():
    """Clear the queue file."""
    save_queue([])


# ---------------------------------------------------------------------------
# Rate limit detection and scheduling
# ---------------------------------------------------------------------------
_RATE_LIMIT_RE = re.compile(
    r"hit your limit.*resets?\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*\(([^)]+)\)",
    re.IGNORECASE,
)

_queue_ready_event = threading.Event()
_queue_timer = None

# Always-on background wake word listener state
_wake_detected = threading.Event()    # bg listener sets when wake word heard
_listener_suppress = False            # True during recording — bg skips predict()
_listener_needs_reset = False         # True after TTS — bg resets its own model


def parse_rate_limit(response):
    """Parse a rate limit message and return a tz-aware datetime for the reset time, or None."""
    m = _RATE_LIMIT_RE.search(response)
    if not m:
        return None
    time_str = m.group(1).strip().lower().replace(" ", "")
    tz_name = m.group(2).strip()
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except (KeyError, zoneinfo.ZoneInfoNotFoundError):
        return None
    now = datetime.now(tz)
    for fmt in ("%I%p", "%I:%M%p"):
        try:
            parsed = datetime.strptime(time_str, fmt)
            reset = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            if reset <= now:
                reset += timedelta(days=1)
            return reset
        except ValueError:
            continue
    return None


def schedule_queue_processing(reset_time):
    """Schedule _queue_ready_event to fire at reset_time."""
    global _queue_timer
    if _queue_timer is not None:
        _queue_timer.cancel()
    if reset_time is None:
        _queue_ready_event.set()
        return
    delay = (reset_time - datetime.now(reset_time.tzinfo)).total_seconds()
    if delay <= 0:
        _queue_ready_event.set()
        return
    _queue_timer = threading.Timer(delay, _queue_ready_event.set)
    _queue_timer.daemon = True
    _queue_timer.start()


def _process_queue(tts_voice, speak_and_clear_fn):
    """Drain the queue, sending each prompt to Claude and speaking the response."""
    q = load_queue()
    if not q:
        return
    speak_and_clear_fn(f"Processing {len(q)} queued prompt{'s' if len(q) != 1 else ''}.")
    while True:
        entry = queue_pop()
        if entry is None:
            break
        prompt = entry["prompt"]
        response = send_to_claude(prompt)
        # Check if we hit rate limit again
        reset_time = parse_rate_limit(response)
        if reset_time is not None:
            queue_add(prompt)
            schedule_queue_processing(reset_time)
            remaining = load_queue()
            delay_min = max(1, int((reset_time - datetime.now(reset_time.tzinfo)).total_seconds() / 60))
            speak_and_clear_fn(
                f"Still rate-limited. I'll retry in about {delay_min} minutes. "
                f"{len(remaining)} prompt{'s' if len(remaining) != 1 else ''} remaining in the queue."
            )
            return
        conversation_logger.info("CLAUDE (queued): %s", response)
        speak_and_clear_fn(response, interruptible=True)
    speak_and_clear_fn("All queued prompts have been processed.")


# ---------------------------------------------------------------------------
# Kokoro TTS wrapper — adapts kokoro_onnx.Kokoro to the interface speak() uses.
# ---------------------------------------------------------------------------
class KokoroTTS:
    """Wrapper around kokoro_onnx.Kokoro that provides a Piper-compatible interface."""

    class _Config:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

    def __init__(self, kokoro, voice_name):
        self._kokoro = kokoro
        self._voice = voice_name
        self.config = self._Config(24000)

    def synthesize(self, text):
        """Yield audio chunks as numpy float32 arrays (matching Piper's interface)."""
        samples, _rate = self._kokoro.create(text, voice=self._voice, speed=1.0)
        yield type('Chunk', (), {'audio_float_array': samples.astype(np.float32)})()


# Lock held while TTS audio is playing, so signal handlers can wait for it.
_tts_lock = threading.Lock()


def speak(tts_voice, text, interrupt_event=None, keep_wake_word=False):
    """Speak text aloud using TTS, streaming via PulseAudio.

    If interrupt_event is provided (a threading.Event), playback stops
    early when the event is set (e.g. by wake word detection).
    If keep_wake_word is True, the wake word is not replaced in the text.
    Returns True if interrupted, False otherwise.
    """
    if not text:
        return False

    debug_log(DEBUG_TTS, f"speak — started ({len(text)} chars): '{text[:60]}'")
    t_speak_start = time.perf_counter()
    text = clean_text_for_speech(text, keep_wake_word=keep_wake_word)
    try:
        player = PulsePlayer(rate=tts_voice.config.sample_rate)
    except RuntimeError as e:
        print(f"ERROR: No audio output device available ({e})")
        print("Exiting — check PulseAudio/PipeWire configuration.")
        os._exit(0)
    interrupted = False
    # Write audio in small sub-chunks so we can check for interrupts frequently.
    # 2048 samples at 22050 Hz ≈ 93ms — responsive enough for wake word interrupts.
    sub_chunk_samples = 2048
    t_first_audio = None
    with _tts_lock:
        try:
            for chunk in tts_voice.synthesize(text):
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                    debug_log(DEBUG_TTS, f"  TTS synthesis to first audio: {t_first_audio - t_speak_start:.3f}s")
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                bytes_per_sub = sub_chunk_samples * 2  # 16-bit = 2 bytes per sample
                for offset in range(0, len(audio_bytes), bytes_per_sub):
                    if interrupt_event and interrupt_event.is_set():
                        interrupted = True
                        break
                    player.write(audio_bytes[offset:offset + bytes_per_sub])
                if interrupted:
                    break
            if not interrupted:
                player.drain()
        finally:
            player.close()
    t_speak_end = time.perf_counter()
    debug_log(DEBUG_TTS, f"speak — finished in {t_speak_end - t_speak_start:.3f}s (interrupted={interrupted})")
    return interrupted


def _always_on_listener(wake_model):
    """Persistent daemon thread that monitors mic for wake word.

    Opens its own PulseAudio recording stream and runs forever. Sets
    _wake_detected when the wake word is heard.  Respects _listener_suppress
    (skips prediction during recording) and _listener_needs_reset (resets
    its own wake model after TTS playback).
    """
    global _listener_suppress, _listener_needs_reset
    try:
        listener = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)
    except RuntimeError as e:
        print(f"ERROR: Listener failed to open audio input ({e})")
        print("Exiting — check PulseAudio/PipeWire configuration.")
        os._exit(0)
    try:
        while True:
            try:
                data = listener.read(CHUNK, exception_on_overflow=False)
            except RuntimeError:
                print("ERROR: Listener audio device disconnected.")
                os._exit(0)

            if _listener_needs_reset:
                reset_wake_model(wake_model)
                _listener_needs_reset = False

            if _listener_suppress or _wake_detected.is_set():
                continue

            audio_data = np.frombuffer(data, dtype=np.int16)
            prediction = wake_model.predict(audio_data)
            for _model_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD:
                    print("\n*** Wake word detected (background listener) ***")
                    _wake_detected.set()
                    break
    finally:
        listener.close()


def _strip_wake_prefix(text):
    """Strip wake word prefix from transcribed text (e.g. 'Hey Jarvis, ...')."""
    prefixes = [
        f"hey {WAKE_WORD_NAME.lower()} ",
        f"hey {WAKE_WORD_NAME.lower()}, ",
        f"{WAKE_WORD_NAME.lower()} ",
        f"{WAKE_WORD_NAME.lower()}, ",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix):
            stripped = text[len(prefix):].strip()
            if stripped:
                return stripped
    return text


SESSION_ID = str(uuid.uuid4())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_logs_dir = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(_logs_dir, exist_ok=True)

error_logger = logging.getLogger("jarvis.error")
error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(os.path.join(_logs_dir, "error.log"))
_error_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
error_logger.addHandler(_error_handler)

conversation_logger = logging.getLogger("jarvis.conversation")
conversation_logger.setLevel(logging.INFO)
_conv_handler = logging.FileHandler(os.path.join(_logs_dir, "conversation.log"))
_conv_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
conversation_logger.addHandler(_conv_handler)


def _tool_status(tool_name, tool_input):
    """Convert a Claude tool_use event into a short speakable status string."""
    if tool_name == "Read":
        fname = os.path.basename(tool_input.get("file_path", "")) if isinstance(tool_input, dict) else ""
        return f"Reading {fname}." if fname else "Reading a file."
    if tool_name in ("Edit", "Write"):
        fname = os.path.basename(tool_input.get("file_path", "")) if isinstance(tool_input, dict) else ""
        return f"Editing {fname}." if fname else "Editing a file."
    if tool_name == "Bash":
        return "Running a command."
    if tool_name in ("Glob", "Grep"):
        return "Searching the codebase."
    if tool_name == "Agent":
        return "Delegating a subtask."
    if tool_name in ("WebFetch", "WebSearch"):
        return "Searching the web."
    return None


def send_to_claude(text, status_queue=None, first_call=[True], proc_holder=None):
    """Send text to Claude Code and return the response, resuming the session.

    If status_queue is provided, pushes short speakable tool-status strings
    as Claude emits tool_use events (stream-json mode).
    If proc_holder is provided (a dict), stores the subprocess as proc_holder["proc"]
    so the caller can kill it mid-processing.
    """
    print(f"\n> {text}\n")
    debug_log(DEBUG_CLAUDE, f"send_to_claude — started, prompt: '{text[:80]}'")
    t_claude_start = time.perf_counter()
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "-p", "--dangerously-skip-permissions",
               "--verbose", "--output-format", "stream-json"]
        if first_call[0]:
            cmd += ["--session-id", SESSION_ID]
            first_call[0] = False
        else:
            cmd += ["--resume", SESSION_ID]
        cmd.append(text)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        if proc_holder is not None:
            proc_holder["proc"] = proc
        t_proc_start = time.perf_counter()
        debug_log(DEBUG_CLAUDE, f"  subprocess started in {t_proc_start - t_claude_start:.3f}s")

        # Timeout watchdog
        timer = threading.Timer(CLAUDE_TIMEOUT, proc.kill)
        timer.start()

        response = None
        t_first_event = None
        event_count = 0
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_count += 1
                if t_first_event is None:
                    t_first_event = time.perf_counter()
                    debug_log(DEBUG_CLAUDE, f"  first event in {t_first_event - t_proc_start:.3f}s (type={event.get('type')})")

                # Extract tool_use events for live status
                if status_queue and event.get("type") == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use":
                            status = _tool_status(block.get("name", ""), block.get("input", {}))
                            if status:
                                debug_log(DEBUG_CLAUDE, f"  tool_use: {block.get('name', '')}")
                                status_queue.put(status)

                # Extract the final result
                if event.get("type") == "result":
                    response = event.get("result", "").strip()

            proc.wait()
        finally:
            timer.cancel()
        t_claude_end = time.perf_counter()
        debug_log(DEBUG_CLAUDE, f"send_to_claude — finished in {t_claude_end - t_claude_start:.3f}s ({event_count} events, response: {len(response) if response else 0} chars)")

        if response is not None:
            if proc.returncode != 0 and not response:
                stderr = proc.stderr.read().strip()
                response = f"Error: {stderr}"
                error_logger.error("Claude returned code %d: %s", proc.returncode, stderr)
            return response

        # Fallback: no result event parsed
        stderr = proc.stderr.read().strip()
        if proc.returncode != 0:
            error_logger.error("Claude returned code %d: %s", proc.returncode, stderr)
            return f"Error: {stderr}" if stderr else "Error: Claude returned no response."
        return "Error: Claude returned no response."

    except FileNotFoundError:
        error_logger.error("'claude' command not found")
        return "Error: 'claude' command not found. Is Claude Code installed?"
    except Exception as e:
        if "timed out" in str(e).lower() or (proc and proc.returncode and proc.returncode < 0):
            error_logger.error("Claude timed out after %d seconds for prompt: %s", CLAUDE_TIMEOUT, text[:200])
            return "Error: Claude Code timed out."
        error_logger.error("Claude error: %s", e)
        return f"Error: {e}"


def main():
    global _listener_suppress, _listener_needs_reset
    wake_model, whisper_model, tts_voice, vad_model = load_models()

    # Load a separate wake word model for the background listener thread.
    # wake_model.predict() has internal state (sliding window) — two threads
    # calling it concurrently corrupts state, so each thread needs its own.
    from openwakeword.model import Model as WakeModel
    if WAKE_WORD_MODEL_PATH:
        _bg_ww_path = WAKE_WORD_MODEL_PATH
    else:
        _bg_ww_path = os.path.join(
            os.path.dirname(__import__('openwakeword').__file__),
            "resources", "models", f"hey_{WAKE_WORD_NAME}_v0.1.onnx")
    bg_wake_model = WakeModel(wakeword_model_paths=[_bg_ww_path])

    # Start always-on background listener
    threading.Thread(
        target=_always_on_listener, args=(bg_wake_model,), daemon=True
    ).start()

    # Start Telegram bot if configured
    if TELEGRAM_TOKEN:
        from telegram_bot import start_telegram_bot
        start_telegram_bot(
            token=TELEGRAM_TOKEN,
            allowed_user_ids=TELEGRAM_ALLOWED_USER_IDS,
            voice_replies=TELEGRAM_VOICE_REPLIES,
            whisper_model=whisper_model,
            tts_voice=tts_voice,
            tts_lock=_tts_lock,
            send_to_claude_fn=send_to_claude,
            conversation_logger=conversation_logger,
        )

    try:
        stream = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)
    except RuntimeError as e:
        print(f"ERROR: No audio input device available ({e})")
        print("Exiting — check PulseAudio/PipeWire configuration.")
        os._exit(0)

    def shutdown(sig, frame):
        print("\nShutting down...")
        speak(tts_voice, random.choice(SHUTDOWN_MESSAGES), keep_wake_word=True)
        os._exit(0)

    def restart(sig, frame):
        print("\nRestarting (waiting for TTS to finish)...")
        _tts_lock.acquire()  # wait for any in-progress speech to complete
        os._exit(42)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGUSR1, restart)

    def speak_and_clear(text, interruptible=False, keep_wake_word=False):
        """Speak text, then flush mic buffer and reset wake model to prevent self-triggering.

        If interruptible=True, the always-on background listener can interrupt
        playback via _wake_detected. Returns True if interrupted, False otherwise.
        If keep_wake_word=True, the wake word is not replaced in the spoken text.
        """
        global _listener_suppress, _listener_needs_reset
        # Track spoken text for echo detection (time-windowed)
        if text:
            _recently_spoken.append((time.time(), text))
        if interruptible:
            _wake_detected.clear()  # clear any stale detection
            interrupted = speak(tts_voice, text, interrupt_event=_wake_detected,
                                keep_wake_word=keep_wake_word)
        else:
            interrupted = speak(tts_voice, text, keep_wake_word=keep_wake_word)
        # After all speech: suppress bg listener, flush main stream, reset models
        _listener_suppress = True
        stream.flush()
        reset_wake_model(wake_model)
        _listener_needs_reset = True   # bg thread resets itself
        _listener_suppress = False
        return interrupted

    wn = WAKE_WORD_NAME.capitalize()
    print(f"\n=== {wn} is ready. Say 'Hey {wn}' to activate. ===\n")
    speak_and_clear(random.choice(STARTUP_MESSAGES), keep_wake_word=True)

    # Notify user if the previous commit was auto-reverted
    revert_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_revert_reason")
    if os.path.exists(revert_marker):
        try:
            reason = open(revert_marker).read().strip()
            os.remove(revert_marker)
            error_logger.error("Auto-reverted commit after preflight failure: %s", reason)
            speak_and_clear(f"Heads up: the last change was automatically reverted because it failed preflight checks. It was: {reason}")
        except OSError:
            pass

    # Check for queued prompts from a previous session
    _startup_queue = load_queue()
    if _startup_queue:
        n = len(_startup_queue)
        print(f"\nQueued prompts ({n}):")
        for i, entry in enumerate(_startup_queue, 1):
            print(f"  {i}. {entry['prompt']}")
        speak_and_clear(f"I have {n} queued prompt{'s' if n != 1 else ''} from before.")
        _queue_ready_event.set()

    skip_wake_word = False
    # Keep a rolling buffer of recent audio chunks so we can capture speech
    # that starts immediately after (or overlapping with) the wake word.
    # 20 chunks × 80ms = 1600ms of pre-roll audio.
    pre_roll_buf = deque(maxlen=20)
    _iter_count = 0
    while True:
        if not skip_wake_word:
            # Read audio chunk for wake word detection
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except RuntimeError:
                print("ERROR: Audio device disconnected.")
                print("Exiting — check PulseAudio/PipeWire configuration.")
                os._exit(0)
            pre_roll_buf.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Feed to wake word detector
            prediction = wake_model.predict(audio_data)

            # Check for wake word activation
            activated = False
            for model_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD:
                    activated = True
                    break
            if not activated:
                # Process queued prompts when the rate limit timer fires
                if _queue_ready_event.is_set():
                    _queue_ready_event.clear()
                    _process_queue(tts_voice, speak_and_clear)
                continue

            _iter_count += 1
            debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} START (wake word detected) ===")
            _iter_start = time.perf_counter()
            print("\n*** Wake word detected! ***")
            # Keep the last 12 chunks (~960ms) of pre-roll. The wake word
            # model has detection latency (~200-300ms after the word ends),
            # so we need enough buffer to capture speech that starts right
            # after the wake word. _strip_wake_prefix() handles any wake
            # word text that ends up in the transcription.
            while len(pre_roll_buf) > 12:
                pre_roll_buf.popleft()
        else:
            _iter_count += 1
            _iter_start = time.perf_counter()
            debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} START (speech interrupt) ===")
            print("\n*** Speech interrupted — listening for command ***")
            skip_wake_word = False

        # Suppress bg listener during recording to avoid concurrent mic access
        _listener_suppress = True
        # Record until silence, prepending buffered audio
        audio_bytes = record_until_silence(stream, vad_model, pre_roll=list(pre_roll_buf))
        pre_roll_buf.clear()

        # Reset wake model and flush mic buffer after every recording
        # to prevent stale audio from re-triggering the wake word
        stream.flush()
        reset_wake_model(wake_model)
        _listener_needs_reset = True
        _listener_suppress = False

        # Transcribe
        with debug_timer(DEBUG_TRANSCRIPTION, "transcribe (end-to-end)"):
            text = transcribe(whisper_model, audio_bytes)
        if not text:
            print("(no speech detected)")
            continue
        if is_garbage_transcription(text):
            debug_log(DEBUG_TRANSCRIPTION, f"filtered garbage transcription: '{text}'")
            print(f"(filtered garbage: {text})")
            continue

        # Strip wake word prefix if the mic captured it in pre-roll audio
        text = _strip_wake_prefix(text)

        # Handle built-in commands before echo filtering — keywords like
        # "restart" are substrings of canned responses ("Restarting now")
        # and would otherwise be incorrectly filtered as self-echo.
        text_lower = text.lower().strip().rstrip(".")
        wn_lower = WAKE_WORD_NAME.lower()
        if text_lower in ("shutdown", "shut down", f"shutdown {wn_lower}",
                          f"shut down {wn_lower}", "turn yourself off",
                          "go to sleep", "goodnight", "good night"):
            print(f"Transcribed: {text}")
            print("Built-in command: shutdown")
            speak(tts_voice, "Shutting down. Goodbye.")
            os._exit(0)

        if text_lower in ("restart", "restart yourself", f"restart {wn_lower}",
                          "please restart", "reboot", "reboot yourself"):
            print(f"Transcribed: {text}")
            print("Built-in command: restart")
            speak(tts_voice, "Restarting now.")
            os._exit(42)

        if text_lower in ("revert", "revert yourself", f"revert {wn_lower}",
                          "revert the last change", "undo the last change",
                          "roll back", "rollback"):
            print(f"Transcribed: {text}")
            print("Built-in command: revert")
            repo_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=repo_dir
            )
            if result.returncode != 0:
                speak_and_clear("Sorry, I couldn't find the git repository.")
                continue
            short_hash = result.stdout.strip()[:7]
            revert = subprocess.run(
                ["git", "revert", "--no-edit", "HEAD"],
                capture_output=True, text=True, cwd=repo_dir
            )
            if revert.returncode != 0:
                speak_and_clear("Sorry, the revert failed.")
                print(f"git revert error: {revert.stderr}")
                continue
            speak(tts_voice, f"Reverted commit {short_hash}. Restarting now.")
            os._exit(42)

        if text_lower in ("queue", "show queue", "whats in the queue",
                          "what's in the queue", "list queue", "pending prompts"):
            print(f"Transcribed: {text}")
            print("Built-in command: queue")
            q = queue_list()
            if not q:
                speak_and_clear("The queue is empty.")
            else:
                parts = [f"There {'is' if len(q) == 1 else 'are'} {len(q)} prompt{'s' if len(q) != 1 else ''} in the queue."]
                for i, entry in enumerate(q, 1):
                    parts.append(f"Number {i}: {entry['prompt']}")
                speak_and_clear(" ".join(parts))
            continue

        if text_lower in ("clear queue", "empty queue", "clear the queue"):
            print(f"Transcribed: {text}")
            print("Built-in command: clear queue")
            global _queue_timer
            queue_clear()
            if _queue_timer is not None:
                _queue_timer.cancel()
                _queue_timer = None
            _queue_ready_event.clear()
            speak_and_clear("Queue cleared.")
            continue

        # Filter out self-echo (mic picking up Jarvis's own speech)
        # If the same echo is repeated multiple times, it's likely intentional.
        global _last_echo_text, _echo_repeat_count
        if is_self_echo(text):
            debug_log(DEBUG_ECHO, f"echo filter: matched as self-echo: '{text[:60]}'")
            norm = _normalize(text)
            if norm == _last_echo_text:
                _echo_repeat_count += 1
            else:
                _last_echo_text = norm
                _echo_repeat_count = 1
            if _echo_repeat_count < ECHO_REPEAT_THRESHOLD:
                print(f"(filtered self-echo: {text})")
                continue
            print(f"(echo repeated {_echo_repeat_count}x — treating as intentional)")
            _last_echo_text = None
            _echo_repeat_count = 0
        else:
            _last_echo_text = None
            _echo_repeat_count = 0

        print(f"Transcribed: {text}")
        conversation_logger.info("USER: %s", text)

        # Send to Claude, with a spoken filler if it takes too long.
        # proc_holder lets us kill Claude mid-processing if user says "stop".
        result_holder = {}
        done_event = threading.Event()
        tool_queue = Queue()
        proc_holder = {}

        def claude_worker():
            result_holder["response"] = send_to_claude(
                text, status_queue=tool_queue, proc_holder=proc_holder
            )
            done_event.set()

        threading.Thread(target=claude_worker, daemon=True).start()

        def _check_wake_during_claude():
            """Handle a wake word detection while Claude is processing.

            Records speech, transcribes it, and decides:
            - "stop"/"cancel"/"never mind"/"nevermind" → return "stop"
            - Non-empty other text → queue it, return "queued"
            - Empty/garbage → return None
            """
            global _listener_suppress, _listener_needs_reset
            _wake_detected.clear()
            _listener_suppress = True
            bg_audio = record_until_silence(stream, vad_model)
            stream.flush()
            reset_wake_model(wake_model)
            _listener_needs_reset = True
            _listener_suppress = False

            bg_text = transcribe(whisper_model, bg_audio)
            if not bg_text or is_garbage_transcription(bg_text):
                return None
            bg_text = _strip_wake_prefix(bg_text)
            bg_lower = bg_text.lower().strip().rstrip(".")
            if bg_lower in ("stop", "cancel", "never mind", "nevermind"):
                return "stop"
            queue_add(bg_text)
            speak_and_clear("Queued. I'll handle that after this task.")
            return "queued"

        stopped = False
        if not done_event.wait(timeout=INITIAL_ACK_DELAY):
            if _wake_detected.is_set():
                wake_result = _check_wake_during_claude()
                if wake_result == "stop":
                    if "proc" in proc_holder:
                        proc_holder["proc"].kill()
                    done_event.wait(timeout=2.0)
                    speak_and_clear("Stopped.")
                    stopped = True
                elif wake_result is None:
                    speak_and_clear(random.choice(ACKNOWLEDGEMENTS))
                # "queued" → continue waiting normally
            else:
                speak_and_clear(random.choice(ACKNOWLEDGEMENTS))

            while not stopped and not done_event.wait(timeout=STILL_WORKING_INTERVAL):
                if _wake_detected.is_set():
                    wake_result = _check_wake_during_claude()
                    if wake_result == "stop":
                        if "proc" in proc_holder:
                            proc_holder["proc"].kill()
                        done_event.wait(timeout=2.0)
                        speak_and_clear("Stopped.")
                        stopped = True
                        break
                    elif wake_result == "queued":
                        continue  # keep waiting for Claude
                    # wake_result is None — fall through to filler
                # Normal filler/tool status logic
                latest_status = None
                try:
                    while True:
                        latest_status = tool_queue.get_nowait()
                except Empty:
                    pass
                if latest_status:
                    speak_and_clear(latest_status)
                else:
                    speak_and_clear(random.choice(STILL_WORKING))

        if stopped:
            continue  # back to top of main loop

        response = result_holder["response"]
        conversation_logger.info("CLAUDE: %s", response)
        print(f"\nClaude: {response}\n")

        # Check for rate limit — queue the prompt and schedule retry
        reset_time = parse_rate_limit(response)
        if reset_time is not None:
            q = queue_add(text)
            schedule_queue_processing(reset_time)
            delay_min = max(1, int((reset_time - datetime.now(reset_time.tzinfo)).total_seconds() / 60))
            speak_and_clear(
                f"You've been rate-limited. I've queued your prompt and will retry in about {delay_min} minutes. "
                f"There {'is' if len(q) == 1 else 'are'} {len(q)} prompt{'s' if len(q) != 1 else ''} in the queue."
            )
            continue

        # Speak response (interruptible by wake word)
        with debug_timer(DEBUG_TTS, "speak response"):
            interrupted = speak_and_clear(response, interruptible=True)
        debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} END — total {time.perf_counter() - _iter_start:.3f}s ===")
        if interrupted:
            skip_wake_word = True


if __name__ == "__main__":
    main()
