"""Load Jarvis configuration from jarvis.ini."""

import configparser
import os

_ini_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jarvis.ini")
_cfg = configparser.ConfigParser()
if not _cfg.read(_ini_path):
    raise FileNotFoundError(f"Config file not found: {_ini_path}")

# Audio config
RATE = _cfg.getint("audio", "rate")
CHANNELS = _cfg.getint("audio", "channels")
CHUNK = _cfg.getint("audio", "chunk")

# Silence detection config (Silero VAD)
VAD_THRESHOLD = _cfg.getfloat("vad", "threshold")
VAD_CHUNK = _cfg.getint("vad", "chunk")
SILENCE_DURATION = _cfg.getfloat("vad", "silence_duration")
PRE_SPEECH_TIMEOUT = _cfg.getfloat("vad", "pre_speech_timeout")
SILENCE_RATIO = _cfg.getfloat("vad", "silence_ratio")
MAX_RECORD_SECONDS = _cfg.getint("vad", "max_record_seconds")

# Wake word detection
WAKE_WORD_THRESHOLD = _cfg.getfloat("wake_word", "threshold")

# Speech-to-text
STT_MODEL = _cfg.get("stt", "model")

# Claude API
CLAUDE_TIMEOUT = _cfg.getint("claude", "timeout")

# TTS config
TTS_ENGINE = _cfg.get("tts", "engine")
TTS_VOICE = _cfg.get("tts", "voice")

# Timing
INITIAL_ACK_DELAY = _cfg.getfloat("timing", "initial_ack_delay")
STILL_WORKING_INTERVAL = _cfg.getint("timing", "still_working_interval")


def _parse_message_list(value):
    """Parse a comma-separated message list from the ini file."""
    return [m.strip() for m in value.split(",") if m.strip()]


# Spoken message lists
ACKNOWLEDGEMENTS = _parse_message_list(_cfg.get("messages", "acknowledgements"))
STILL_WORKING = _parse_message_list(_cfg.get("messages", "still_working"))
STARTUP_MESSAGES = _parse_message_list(_cfg.get("messages", "startup"))
SHUTDOWN_MESSAGES = _parse_message_list(_cfg.get("messages", "shutdown"))
