#!/usr/bin/env python3
"""Jarvis: Wake-word voice assistant for Claude Code."""

import ctypes
from collections import deque
from datetime import datetime
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
import wave

# Suppress onnxruntime CUDA warning
import warnings
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import numpy as np

# Audio config
RATE = 16000
CHANNELS = 1
CHUNK = 1280  # 80ms at 16kHz - openwakeword expects this

# PulseAudio simple API via ctypes
_pulse_simple = ctypes.cdll.LoadLibrary("libpulse-simple.so.0")

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

    def close(self):
        if self._pa:
            _pulse_simple.pa_simple_free(self._pa)
            self._pa = None

# Silence detection config
NOISE_MULTIPLIER = 3.0         # speech threshold = ambient_rms * this
SILENCE_DURATION = 1.5         # seconds of silence after speech to stop
SILENCE_RATIO = 0.8            # fraction of silence window that must be quiet
MAX_RECORD_SECONDS = 30        # safety cap

# Spoken acknowledgements while waiting for Claude API response
ACKNOWLEDGEMENTS = [
    "Let me think about that.",
    "One moment.",
    "Working on it.",
    "Give me a second.",
    "On it.",
    "Let me look into that.",
    "Hmm, let me see.",
    "Just a moment.",
]

# Secondary acknowledgements for long-running requests (spoken every few seconds)
STILL_WORKING = [
    "Still working on it.",
    "Almost there.",
    "Still thinking.",
    "Hang on, still going.",
    "Bear with me.",
    "Still on it.",
]

STILL_WORKING_INTERVAL = 5  # seconds between secondary acknowledgements


def load_models():
    """Load wake word and whisper models."""
    print("Loading wake word model...")
    from openwakeword.model import Model as WakeModel
    wake_model = WakeModel(wakeword_model_paths=[
        os.path.join(os.path.dirname(__import__('openwakeword').__file__),
                     "resources", "models", "hey_jarvis_v0.1.onnx")
    ])

    print("Loading whisper model (small.en)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    print("Loading TTS model...")
    from piper import PiperVoice
    # Download voice if needed
    voice_dir = os.path.join(os.path.dirname(__file__), "voices")
    voice_path = os.path.join(voice_dir, "en_US-lessac-medium.onnx")
    voice_config = voice_path + ".json"

    if not os.path.exists(voice_path):
        os.makedirs(voice_dir, exist_ok=True)
        print("Downloading TTS voice...")
        import urllib.request
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
        urllib.request.urlretrieve(f"{base}/en_US-lessac-medium.onnx", voice_path)
        urllib.request.urlretrieve(f"{base}/en_US-lessac-medium.onnx.json", voice_config)

    tts_voice = PiperVoice.load(voice_path)

    print("All models loaded.")
    return wake_model, whisper_model, tts_voice


class NoiseTracker:
    """Continuously tracks ambient noise using an exponential moving average."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.ambient_rms = None

    def update(self, rms):
        if self.ambient_rms is None:
            self.ambient_rms = rms
        else:
            self.ambient_rms = self.alpha * rms + (1 - self.alpha) * self.ambient_rms

    @property
    def speech_threshold(self):
        if self.ambient_rms is None:
            return 200
        return max(self.ambient_rms * NOISE_MULTIPLIER, 200)


def record_until_silence(stream, noise_tracker):
    """Record audio until speech is followed by silence. Returns raw audio bytes."""
    print("Listening...")
    frames = []
    speech_detected = False
    chunks_per_second = RATE / CHUNK
    silence_window_size = int(SILENCE_DURATION * chunks_per_second)

    # Rolling window: track whether each recent chunk was silent
    window = deque(maxlen=silence_window_size)

    for _ in range(int(MAX_RECORD_SECONDS * chunks_per_second)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_data ** 2))

        is_silent = rms < noise_tracker.speech_threshold

        if not is_silent:
            speech_detected = True

        if speech_detected:
            window.append(is_silent)

        # Stop when the rolling window is full and mostly silent
        if speech_detected and len(window) == silence_window_size:
            if sum(window) / silence_window_size >= SILENCE_RATIO:
                break

    print("Done recording.")
    return b"".join(frames)


def transcribe(whisper_model, audio_bytes):
    """Transcribe raw audio bytes with faster-whisper."""
    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)

    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    finally:
        os.unlink(tmp_path)


def speak(tts_voice, text):
    """Speak text aloud using piper TTS."""
    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(tts_voice.config.sample_rate)
            for chunk in tts_voice.synthesize(text):
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

    try:
        subprocess.run(["aplay", "-q", tmp_path], check=False)
    finally:
        os.unlink(tmp_path)


SESSION_ID = str(uuid.uuid4())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

error_logger = logging.getLogger("jarvis.error")
error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(os.path.join(SCRIPT_DIR, "error.log"))
_error_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
error_logger.addHandler(_error_handler)

conversation_logger = logging.getLogger("jarvis.conversation")
conversation_logger.setLevel(logging.INFO)
_conv_handler = logging.FileHandler("conversation.log")
_conv_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
conversation_logger.addHandler(_conv_handler)


def send_to_claude(text, first_call=[True]):
    """Send text to Claude Code and return the response, resuming the session."""
    print(f"\n> {text}\n")
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "-p", "--dangerously-skip-permissions"]
        if first_call[0]:
            cmd += ["--session-id", SESSION_ID]
            first_call[0] = False
        else:
            cmd += ["--resume", SESSION_ID]
        cmd.append(text)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        response = result.stdout.strip()
        if result.returncode != 0 and not response:
            response = f"Error: {result.stderr.strip()}"
            error_logger.error("Claude returned code %d: %s", result.returncode, result.stderr.strip())
        return response
    except FileNotFoundError:
        error_logger.error("'claude' command not found")
        return "Error: 'claude' command not found. Is Claude Code installed?"
    except subprocess.TimeoutExpired:
        error_logger.error("Claude timed out after 120 seconds for prompt: %s", text[:200])
        return "Error: Claude Code timed out."


def main():
    wake_model, whisper_model, tts_voice = load_models()

    stream = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)

    noise_tracker = NoiseTracker()

    def shutdown(sig, frame):
        print("\nShutting down...")
        os._exit(0)

    def restart(sig, frame):
        print("\nRestarting...")
        os._exit(42)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGUSR1, restart)

    print("\n=== Jarvis is ready. Say 'Hey Jarvis' to activate. ===\n")

    while True:
        # Read audio chunk for wake word detection
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Continuously track ambient noise (only during non-speech)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        if rms < noise_tracker.speech_threshold:
            noise_tracker.update(rms)

        # Feed to wake word detector
        prediction = wake_model.predict(audio_data)

        # Check for wake word activation
        for model_name, score in prediction.items():
            if score > 0.7:
                print(f"\n*** Wake word detected! (threshold: {noise_tracker.speech_threshold:.0f}) ***")

                # Record until silence
                audio_bytes = record_until_silence(stream, noise_tracker)

                # Transcribe
                text = transcribe(whisper_model, audio_bytes)
                if not text:
                    print("(no speech detected)")
                    continue

                print(f"Transcribed: {text}")
                conversation_logger.info("USER: %s", text)

                # Handle built-in commands without calling Claude API
                text_lower = text.lower().strip().rstrip(".")
                if text_lower in ("restart", "restart yourself", "restart jarvis",
                                  "please restart", "reboot", "reboot yourself"):
                    print("Built-in command: restart")
                    speak(tts_voice, "Restarting now.")
                    os._exit(42)

                if text_lower in ("revert", "revert yourself", "revert jarvis",
                                  "revert the last change", "undo the last change",
                                  "roll back", "rollback"):
                    print("Built-in command: revert")
                    repo_dir = os.path.dirname(os.path.abspath(__file__))
                    result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True, text=True, cwd=repo_dir
                    )
                    if result.returncode != 0:
                        speak(tts_voice, "Sorry, I couldn't find the git repository.")
                        wake_model.reset()
                        continue
                    short_hash = result.stdout.strip()[:7]
                    revert = subprocess.run(
                        ["git", "revert", "--no-edit", "HEAD"],
                        capture_output=True, text=True, cwd=repo_dir
                    )
                    if revert.returncode != 0:
                        speak(tts_voice, "Sorry, the revert failed.")
                        print(f"git revert error: {revert.stderr}")
                        wake_model.reset()
                        continue
                    speak(tts_voice, f"Reverted commit {short_hash}. Restarting now.")
                    os._exit(42)

                # Send to Claude, with a spoken filler if it takes too long
                result_holder = {}
                done_event = threading.Event()

                def claude_worker():
                    result_holder["response"] = send_to_claude(text)
                    done_event.set()

                threading.Thread(target=claude_worker, daemon=True).start()

                if not done_event.wait(timeout=1.0):
                    speak(tts_voice, random.choice(ACKNOWLEDGEMENTS))
                    while not done_event.wait(timeout=STILL_WORKING_INTERVAL):
                        speak(tts_voice, random.choice(STILL_WORKING))

                response = result_holder["response"]
                conversation_logger.info("CLAUDE: %s", response)
                print(f"\nClaude: {response}\n")

                # Speak response
                speak(tts_voice, response)

                # Reset wake word model state
                wake_model.reset()


if __name__ == "__main__":
    main()
