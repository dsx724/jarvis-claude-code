#!/usr/bin/env python3
"""Jarvis: Wake-word voice assistant for Claude Code."""

import ctypes
from collections import deque
from contextlib import contextmanager
from datetime import datetime
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

# Suppress onnxruntime CUDA warning
import warnings
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import numpy as np

# ---------------------------------------------------------------------------
# Load configuration from config/jarvis.ini
# ---------------------------------------------------------------------------
import configparser

_ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "jarvis.ini")
_cfg = configparser.ConfigParser()
if not _cfg.read(_ini_path):
    raise FileNotFoundError(f"Config file not found: {_ini_path}")

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
        handler = logging.FileHandler(os.path.join(_script_dir, "logs", "debug.log"))
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
STT_MODEL = _cfg.get("stt", "model")
CLAUDE_TIMEOUT = _cfg.getint("claude", "timeout")
TTS_ENGINE = _cfg.get("tts", "engine")
TTS_VOICE = _cfg.get("tts", "voice")
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


def load_models():
    """Load wake word and whisper models."""
    with debug_timer(DEBUG_MODELS, "load_models total"):
        print("Loading wake word model...")
        with debug_timer(DEBUG_MODELS, "load wake word model"):
            from openwakeword.model import Model as WakeModel
            wake_model = WakeModel(wakeword_model_paths=[
                os.path.join(os.path.dirname(__import__('openwakeword').__file__),
                             "resources", "models", f"hey_{WAKE_WORD_NAME}_v0.1.onnx")
            ])

        print(f"Loading whisper model ({STT_MODEL})...")
        with debug_timer(DEBUG_MODELS, "load whisper model"):
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")

        print("Loading Silero VAD model...")
        with debug_timer(DEBUG_MODELS, "load Silero VAD model"):
            from silero_vad import load_silero_vad
            vad_model = load_silero_vad(onnx=True)

        print(f"Loading TTS model ({TTS_ENGINE}: {TTS_VOICE})...")
        with debug_timer(DEBUG_MODELS, "load TTS model"):
            if TTS_ENGINE != "piper":
                raise ValueError(f"Unsupported TTS engine: {TTS_ENGINE}")
            from piper import PiperVoice
            # Download voice if needed
            voice_dir = os.path.join(os.path.dirname(__file__), "voices")
            voice_path = os.path.join(voice_dir, f"{TTS_VOICE}.onnx")
            voice_config = voice_path + ".json"

            if not os.path.exists(voice_path):
                os.makedirs(voice_dir, exist_ok=True)
                print(f"Downloading TTS voice ({TTS_VOICE})...")
                import urllib.request
                # Parse voice name: en_US-lessac-medium -> en/en_US/lessac/medium
                parts = TTS_VOICE.split("-")
                lang = parts[0]                    # en_US
                lang_family = lang.split("_")[0]   # en
                dataset = parts[1]                 # lessac
                quality = parts[2]                 # medium
                base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang}/{dataset}/{quality}"
                urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx", voice_path)
                urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx.json", voice_config)

            tts_voice = PiperVoice.load(voice_path)

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
        data = stream.read(VAD_CHUNK, exception_on_overflow=False)
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
    """Transcribe raw audio bytes with faster-whisper."""
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
        segments, info = whisper_model.transcribe(
            tmp_path, beam_size=1, without_timestamps=True, vad_filter=True
        )
        t_transcribe_start = time.perf_counter()
        debug_log(DEBUG_TRANSCRIPTION, f"  whisper.transcribe() call: {t_transcribe_start - t_wav:.3f}s")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        t_segments = time.perf_counter()
        debug_log(DEBUG_TRANSCRIPTION, f"  segment iteration: {t_segments - t_transcribe_start:.3f}s")
        debug_log(DEBUG_TRANSCRIPTION, f"transcribe — finished in {t_segments - t_start:.3f}s, result: '{text[:80]}'")
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
    # Replace wake word so TTS doesn't trigger the wake word detector
    text = re.sub(r'\b' + re.escape(WAKE_WORD_NAME) + r'\b', 'wake word', text, flags=re.IGNORECASE)
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
    """Return set of normalized canned messages (built once)."""
    global _canned_normalized
    if _canned_normalized is None:
        _canned_normalized = {_normalize(m) for m in _CANNED_MESSAGES}
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


# Lock held while TTS audio is playing, so signal handlers can wait for it.
_tts_lock = threading.Lock()


def speak(tts_voice, text, interrupt_event=None):
    """Speak text aloud using piper TTS, streaming via PulseAudio.

    If interrupt_event is provided (a threading.Event), playback stops
    early when the event is set (e.g. by wake word detection).
    Returns True if interrupted, False otherwise.
    """
    if not text:
        return False

    debug_log(DEBUG_TTS, f"speak — started ({len(text)} chars): '{text[:60]}'")
    t_speak_start = time.perf_counter()
    text = clean_text_for_speech(text)
    player = PulsePlayer(rate=tts_voice.config.sample_rate)
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


def send_to_claude(text, status_queue=None, first_call=[True]):
    """Send text to Claude Code and return the response, resuming the session.

    If status_queue is provided, pushes short speakable tool-status strings
    as Claude emits tool_use events (stream-json mode).
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
    wake_model, whisper_model, tts_voice, vad_model = load_models()

    stream = PulseRecorder(rate=RATE, channels=CHANNELS, chunk=CHUNK)

    def shutdown(sig, frame):
        print("\nShutting down...")
        speak(tts_voice, random.choice(SHUTDOWN_MESSAGES))
        os._exit(0)

    def restart(sig, frame):
        print("\nRestarting (waiting for TTS to finish)...")
        _tts_lock.acquire()  # wait for any in-progress speech to complete
        os._exit(42)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGUSR1, restart)

    def speak_and_clear(text, interruptible=False):
        """Speak text, then flush mic buffer and reset wake model to prevent self-triggering.

        If interruptible=True, listens for the wake word during playback and
        stops early if detected. Returns True if interrupted, False otherwise.
        """
        # Track spoken text for echo detection (time-windowed)
        if text:
            _recently_spoken.append((time.time(), text))
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
        reset_wake_model(wake_model)
        return interrupted

    wn = WAKE_WORD_NAME.capitalize()
    print(f"\n=== {wn} is ready. Say 'Hey {wn}' to activate. ===\n")
    speak_and_clear(random.choice(STARTUP_MESSAGES))

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

    skip_wake_word = False
    # Keep a rolling buffer of recent audio chunks so we can capture speech
    # that starts immediately after (or overlapping with) the wake word.
    # 8 chunks × 80ms = 640ms of pre-roll audio.
    pre_roll_buf = deque(maxlen=8)
    _iter_count = 0
    while True:
        if not skip_wake_word:
            # Read audio chunk for wake word detection
            data = stream.read(CHUNK, exception_on_overflow=False)
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
                continue

            _iter_count += 1
            debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} START (wake word detected) ===")
            _iter_start = time.perf_counter()
            print("\n*** Wake word detected! ***")
            # Keep only the last 2 chunks (~160ms) — recent enough to capture
            # speech that starts right after the wake word, but discards the
            # earlier chunks that contain the wake word itself.
            while len(pre_roll_buf) > 2:
                pre_roll_buf.popleft()
        else:
            _iter_count += 1
            _iter_start = time.perf_counter()
            debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} START (speech interrupt) ===")
            print("\n*** Speech interrupted — listening for command ***")
            skip_wake_word = False

        # Record until silence, prepending buffered audio
        audio_bytes = record_until_silence(stream, vad_model, pre_roll=list(pre_roll_buf))
        pre_roll_buf.clear()

        # Reset wake model and flush mic buffer after every recording
        # to prevent stale audio from re-triggering the wake word
        stream.flush()
        reset_wake_model(wake_model)

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

        # Handle built-in commands before echo filtering — keywords like
        # "restart" are substrings of canned responses ("Restarting now")
        # and would otherwise be incorrectly filtered as self-echo.
        text_lower = text.lower().strip().rstrip(".")
        wn_lower = WAKE_WORD_NAME.lower()
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

        # Send to Claude, with a spoken filler if it takes too long
        result_holder = {}
        done_event = threading.Event()
        tool_queue = Queue()

        def claude_worker():
            result_holder["response"] = send_to_claude(text, status_queue=tool_queue)
            done_event.set()

        threading.Thread(target=claude_worker, daemon=True).start()

        if not done_event.wait(timeout=INITIAL_ACK_DELAY):
            speak_and_clear(random.choice(ACKNOWLEDGEMENTS))
            while not done_event.wait(timeout=STILL_WORKING_INTERVAL):
                # Drain queue, keep only the latest tool status
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

        response = result_holder["response"]
        conversation_logger.info("CLAUDE: %s", response)
        print(f"\nClaude: {response}\n")

        # Speak response (interruptible by wake word)
        with debug_timer(DEBUG_TTS, "speak response"):
            interrupted = speak_and_clear(response, interruptible=True)
        debug_log(DEBUG_RECORDING, f"=== iteration {_iter_count} END — total {time.perf_counter() - _iter_start:.3f}s ===")
        if interrupted:
            skip_wake_word = True


if __name__ == "__main__":
    main()
