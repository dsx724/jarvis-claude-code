#!/usr/bin/env python3
"""Jarvis: Wake-word voice assistant for Claude Code."""

import ctypes
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
import logging
import os
import random
import re
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RATE, CHANNELS, CHUNK,
    VAD_THRESHOLD, VAD_CHUNK, SILENCE_DURATION, SILENCE_RATIO, MAX_RECORD_SECONDS,
    WAKE_WORD_THRESHOLD, CLAUDE_TIMEOUT, INITIAL_ACK_DELAY,
    ACKNOWLEDGEMENTS, STILL_WORKING, STILL_WORKING_INTERVAL,
    STARTUP_MESSAGES, SHUTDOWN_MESSAGES,
)

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


def load_models():
    """Load wake word and whisper models."""
    print("Loading wake word model...")
    from openwakeword.model import Model as WakeModel
    wake_model = WakeModel(wakeword_model_paths=[
        os.path.join(os.path.dirname(__import__('openwakeword').__file__),
                     "resources", "models", "hey_jarvis_v0.1.onnx")
    ])

    print("Loading whisper model (medium.en)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("medium.en", device="cpu", compute_type="int8")

    print("Loading Silero VAD model...")
    from silero_vad import load_silero_vad
    vad_model = load_silero_vad(onnx=True)

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
    return wake_model, whisper_model, tts_voice, vad_model


def record_until_silence(stream, vad_model):
    """Record audio until speech is followed by silence using Silero VAD. Returns raw audio bytes."""
    import torch

    print("Listening...")
    vad_model.reset_states()
    frames = []
    speech_detected = False
    chunks_per_second = RATE / VAD_CHUNK
    silence_window_size = int(SILENCE_DURATION * chunks_per_second)

    # Rolling window: track whether each recent chunk was silent
    window = deque(maxlen=silence_window_size)

    for _ in range(int(MAX_RECORD_SECONDS * chunks_per_second)):
        data = stream.read(VAD_CHUNK, exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(audio_data)
        speech_prob = vad_model(chunk_tensor, RATE).item()

        is_silent = speech_prob < VAD_THRESHOLD

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


def clean_text_for_speech(text):
    """Strip markdown formatting that TTS would speak literally."""
    import re
    # Remove bold/italic markers (**, *, __, _)
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'_{1,3}', '', text)
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove inline code backticks
    text = re.sub(r'`{1,3}', '', text)
    # Remove bullet point markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    return text.strip()


# Track recently spoken text for echo detection
_recently_spoken = []
ECHO_SIMILARITY_THRESHOLD = 0.5


def _normalize(text):
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def is_self_echo(transcribed):
    """Check if transcribed text is an echo of something Jarvis recently said."""
    norm_t = _normalize(transcribed)
    if not norm_t:
        return False
    for spoken in _recently_spoken:
        norm_s = _normalize(spoken)
        if not norm_s:
            continue
        # Check if transcription is a substring of spoken text
        if norm_t in norm_s:
            return True
        # Check fuzzy similarity
        ratio = SequenceMatcher(None, norm_t, norm_s).ratio()
        if ratio >= ECHO_SIMILARITY_THRESHOLD:
            return True
    return False


def speak(tts_voice, text, interrupt_event=None):
    """Speak text aloud using piper TTS, streaming via PulseAudio.

    If interrupt_event is provided (a threading.Event), playback stops
    early when the event is set (e.g. by wake word detection).
    Returns True if interrupted, False otherwise.
    """
    if not text:
        return False

    text = clean_text_for_speech(text)
    player = PulsePlayer(rate=tts_voice.config.sample_rate)
    interrupted = False
    try:
        for chunk in tts_voice.synthesize(text):
            if interrupt_event and interrupt_event.is_set():
                interrupted = True
                break
            audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
            player.write(audio_int16.tobytes())
        if not interrupted:
            player.drain()
    finally:
        player.close()
    return interrupted


def listen_for_wake_word(wake_model, interrupt_event):
    """Monitor mic for wake word in a background thread, setting interrupt_event when detected.

    Opens its own PulseAudio recording stream so it can listen independently
    of the main stream.
    """
    listener = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)
    try:
        while not interrupt_event.is_set():
            data = listener.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            prediction = wake_model.predict(audio_data)
            for model_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD:
                    print("\n*** Wake word detected (interrupting speech) ***")
                    interrupt_event.set()
                    break
    finally:
        listener.close()


SESSION_ID = str(uuid.uuid4())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

error_logger = logging.getLogger("jarvis.error")
error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(os.path.join(SCRIPT_DIR, "logs", "error.log"))
_error_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
error_logger.addHandler(_error_handler)

conversation_logger = logging.getLogger("jarvis.conversation")
conversation_logger.setLevel(logging.INFO)
_conv_handler = logging.FileHandler(os.path.join(SCRIPT_DIR, "logs", "conversation.log"))
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
            timeout=CLAUDE_TIMEOUT,
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
        error_logger.error("Claude timed out after %d seconds for prompt: %s", CLAUDE_TIMEOUT, text[:200])
        return "Error: Claude Code timed out."


def main():
    wake_model, whisper_model, tts_voice, vad_model = load_models()

    stream = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)

    def shutdown(sig, frame):
        print("\nShutting down...")
        speak(tts_voice, random.choice(SHUTDOWN_MESSAGES))
        os._exit(0)

    def restart(sig, frame):
        print("\nRestarting...")
        os._exit(42)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGUSR1, restart)

    def speak_and_clear(text, interruptible=False):
        """Speak text, then flush mic buffer and reset wake model to prevent self-triggering.

        If interruptible=True, listens for the wake word during playback and
        stops early if detected. Returns True if interrupted, False otherwise.
        """
        # Track spoken text for echo detection (keep last 3)
        if text:
            _recently_spoken.append(text)
            if len(_recently_spoken) > 3:
                _recently_spoken.pop(0)
        if interruptible:
            interrupt_event = threading.Event()
            listener_thread = threading.Thread(
                target=listen_for_wake_word,
                args=(wake_model, interrupt_event),
                daemon=True,
            )
            listener_thread.start()
            interrupted = speak(tts_voice, text, interrupt_event=interrupt_event)
            interrupt_event.set()  # signal listener to stop if still running
            listener_thread.join(timeout=1.0)
        else:
            interrupted = speak(tts_voice, text)
        stream.flush()
        wake_model.reset()
        return interrupted

    print("\n=== Jarvis is ready. Say 'Hey Jarvis' to activate. ===\n")
    speak_and_clear(random.choice(STARTUP_MESSAGES))

    skip_wake_word = False
    while True:
        if not skip_wake_word:
            # Read audio chunk for wake word detection
            data = stream.read(CHUNK, exception_on_overflow=False)
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
                continue

            print("\n*** Wake word detected! ***")
        else:
            print("\n*** Speech interrupted — listening for command ***")
            skip_wake_word = False

        # Record until silence
        audio_bytes = record_until_silence(stream, vad_model)

        # Transcribe
        text = transcribe(whisper_model, audio_bytes)
        if not text:
            print("(no speech detected)")
            continue

        # Filter out self-echo (mic picking up Jarvis's own speech)
        if is_self_echo(text):
            print(f"(filtered self-echo: {text})")
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

        # Send to Claude, with a spoken filler if it takes too long
        result_holder = {}
        done_event = threading.Event()

        def claude_worker():
            result_holder["response"] = send_to_claude(text)
            done_event.set()

        threading.Thread(target=claude_worker, daemon=True).start()

        if not done_event.wait(timeout=INITIAL_ACK_DELAY):
            speak_and_clear(random.choice(ACKNOWLEDGEMENTS))
            while not done_event.wait(timeout=STILL_WORKING_INTERVAL):
                speak_and_clear(random.choice(STILL_WORKING))

        response = result_holder["response"]
        conversation_logger.info("CLAUDE: %s", response)
        print(f"\nClaude: {response}\n")

        # Speak response (interruptible by wake word)
        interrupted = speak_and_clear(response, interruptible=True)
        if interrupted:
            skip_wake_word = True


if __name__ == "__main__":
    main()
