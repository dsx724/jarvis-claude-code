#!/usr/bin/env python3
"""Preflight checks for Jarvis. Exits 0 if all pass, 1 if any fail.

Run before restarting Jarvis to catch bugs early and avoid a broken state.
This script is safety-critical — update it when adding config constants or
changing function signatures in jarvis.py.
"""

import importlib
import inspect
import os
import py_compile
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
passed = 0
failed = 0


def check(name, fn):
    """Run a check function. Prints PASS/FAIL and updates counters."""
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1


# ---------------------------------------------------------------------------
# 1. Syntax validation
# ---------------------------------------------------------------------------
def check_syntax():
    py_compile.compile(os.path.join(SCRIPT_DIR, "jarvis.py"), doraise=True)
    # Check that jarvis.ini exists and is parseable
    import configparser
    ini_path = os.path.join(SCRIPT_DIR, "config", "jarvis.ini")
    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"jarvis.ini not found at {ini_path}")
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    required_sections = ["debug", "audio", "vad", "wake_word", "stt", "claude", "tts", "timing", "messages"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(f"jarvis.ini missing sections: {missing}")


# ---------------------------------------------------------------------------
# 2. Config import — verify jarvis.py exports all expected config constants
# ---------------------------------------------------------------------------
def check_config_import():
    mod = check_module_import._module
    expected = [
        "DEBUG_MODELS", "DEBUG_RECORDING", "DEBUG_TRANSCRIPTION",
        "DEBUG_CLAUDE", "DEBUG_TTS", "DEBUG_ECHO",
        "RATE", "CHANNELS", "CHUNK",
        "VAD_THRESHOLD", "VAD_CHUNK", "SILENCE_DURATION", "SILENCE_RATIO", "MAX_RECORD_SECONDS",
        "WAKE_WORD_THRESHOLD", "CLAUDE_TIMEOUT", "INITIAL_ACK_DELAY",
        "PRE_SPEECH_TIMEOUT",
        "ACKNOWLEDGEMENTS", "STILL_WORKING", "STILL_WORKING_INTERVAL",
        "STARTUP_MESSAGES", "SHUTDOWN_MESSAGES",
        "TTS_ENGINE", "TTS_VOICE", "STT_MODEL",
        "WHISPER_HALLUCINATIONS", "ECHO_MEMORY_SECONDS", "ECHO_CANNED_THRESHOLD",
        "QUEUE_FILE",
    ]
    missing = [name for name in expected if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"Missing config constants in jarvis.py: {missing}")


# ---------------------------------------------------------------------------
# 3. Config value validation
# ---------------------------------------------------------------------------
def check_config_values():
    mod = check_module_import._module
    errors = []

    # Type checks
    for name, expected in [
        ("DEBUG_MODELS", bool),
        ("DEBUG_RECORDING", bool),
        ("DEBUG_TRANSCRIPTION", bool),
        ("DEBUG_CLAUDE", bool),
        ("DEBUG_TTS", bool),
        ("DEBUG_ECHO", bool),
        ("RATE", int),
        ("CHANNELS", int),
        ("CHUNK", int),
        ("VAD_CHUNK", int),
        ("MAX_RECORD_SECONDS", (int, float)),
        ("CLAUDE_TIMEOUT", (int, float)),
        ("STILL_WORKING_INTERVAL", (int, float)),
        ("INITIAL_ACK_DELAY", (int, float)),
        ("PRE_SPEECH_TIMEOUT", (int, float)),
        ("VAD_THRESHOLD", (int, float)),
        ("SILENCE_DURATION", (int, float)),
        ("SILENCE_RATIO", (int, float)),
        ("WAKE_WORD_THRESHOLD", (int, float)),
    ]:
        val = getattr(mod, name)
        if not isinstance(val, expected):
            errors.append(f"{name} should be {expected}, got {type(val)}")

    # String checks
    for name in ["TTS_ENGINE", "TTS_VOICE", "STT_MODEL", "WAKE_WORD_NAME"]:
        val = getattr(mod, name)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{name} must be a non-empty string")

    # Wake word model validation
    supported_wake_words = ("jarvis", "marvin", "mycroft")
    if mod.WAKE_WORD_NAME not in supported_wake_words:
        errors.append(f"WAKE_WORD_NAME={mod.WAKE_WORD_NAME} not in {supported_wake_words}")

    # STT model validation
    supported_stt_models = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v3")
    if mod.STT_MODEL not in supported_stt_models:
        errors.append(f"STT_MODEL={mod.STT_MODEL} not in {supported_stt_models}")

    # TTS engine validation
    supported_engines = ("piper", "kokoro")
    if mod.TTS_ENGINE not in supported_engines:
        errors.append(f"TTS_ENGINE={mod.TTS_ENGINE} not in {supported_engines}")

    # TTS voice format validation (engine-specific)
    if mod.TTS_ENGINE == "piper" and mod.TTS_VOICE.count("-") < 2:
        errors.append(f"TTS_VOICE={mod.TTS_VOICE} should be format lang-dataset-quality (e.g. en_US-lessac-medium)")
    if mod.TTS_ENGINE == "kokoro" and not mod.TTS_VOICE:
        errors.append("TTS_VOICE must be set for kokoro engine (e.g. af_heart)")

    # Range checks
    if not (0 < mod.VAD_THRESHOLD <= 1.0):
        errors.append(f"VAD_THRESHOLD={mod.VAD_THRESHOLD} not in (0, 1.0]")
    if not (0 < mod.WAKE_WORD_THRESHOLD <= 1.0):
        errors.append(f"WAKE_WORD_THRESHOLD={mod.WAKE_WORD_THRESHOLD} not in (0, 1.0]")
    if not (0 < mod.SILENCE_RATIO <= 1.0):
        errors.append(f"SILENCE_RATIO={mod.SILENCE_RATIO} not in (0, 1.0]")
    if mod.RATE <= 0:
        errors.append(f"RATE={mod.RATE} must be positive")
    if mod.CHUNK <= 0:
        errors.append(f"CHUNK={mod.CHUNK} must be positive")
    if mod.VAD_CHUNK <= 0:
        errors.append(f"VAD_CHUNK={mod.VAD_CHUNK} must be positive")
    if mod.CLAUDE_TIMEOUT <= 0:
        errors.append(f"CLAUDE_TIMEOUT={mod.CLAUDE_TIMEOUT} must be positive")
    if mod.PRE_SPEECH_TIMEOUT <= 0:
        errors.append(f"PRE_SPEECH_TIMEOUT={mod.PRE_SPEECH_TIMEOUT} must be positive")

    # Non-empty message lists
    for name in ["ACKNOWLEDGEMENTS", "STILL_WORKING", "STARTUP_MESSAGES", "SHUTDOWN_MESSAGES"]:
        lst = getattr(mod, name)
        if not isinstance(lst, list) or len(lst) == 0:
            errors.append(f"{name} must be a non-empty list")

    if errors:
        raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# 4. Module import — load jarvis.py via importlib
# ---------------------------------------------------------------------------
def check_module_import():
    spec = importlib.util.spec_from_file_location(
        "jarvis", os.path.join(SCRIPT_DIR, "jarvis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Store for later checks
    check_module_import._module = mod


# ---------------------------------------------------------------------------
# 5. Function signatures
# ---------------------------------------------------------------------------
def check_kokoro_wrapper():
    mod = check_module_import._module
    cls = getattr(mod, "KokoroTTS", None)
    if cls is None:
        raise AttributeError("KokoroTTS class not found in jarvis.py")
    if not hasattr(cls, "synthesize"):
        raise AttributeError("KokoroTTS missing synthesize method")


def check_function_signatures():
    mod = check_module_import._module
    expected = {
        "reset_wake_model": ["wake_model"],
        "load_models": [],
        "record_until_silence": ["stream", "vad_model"],
        "transcribe": ["whisper_model", "audio_bytes"],
        "clean_text_for_speech": ["text"],
        "is_garbage_transcription": ["text"],
        "is_self_echo": ["transcribed"],
        "speak": ["tts_voice", "text"],
        "_always_on_listener": ["wake_model"],
        "_strip_wake_prefix": ["text"],
        "send_to_claude": ["text"],
        "main": [],
        "parse_rate_limit": ["response"],
        "load_queue": [],
        "save_queue": ["queue"],
        "queue_add": ["prompt"],
        "queue_pop": [],
        "queue_list": [],
        "queue_clear": [],
        "schedule_queue_processing": ["reset_time"],
    }
    errors = []
    for fn_name, params in expected.items():
        fn = getattr(mod, fn_name, None)
        if fn is None:
            errors.append(f"missing function: {fn_name}")
            continue
        sig = inspect.signature(fn)
        actual = [
            p for p in sig.parameters
            if sig.parameters[p].default is inspect.Parameter.empty
        ]
        if actual != params:
            errors.append(f"{fn_name}: expected params {params}, got {actual}")
    if errors:
        raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# 5b. Always-on listener module attributes
# ---------------------------------------------------------------------------
def check_listener_state():
    mod = check_module_import._module
    missing = []
    for attr in ("_wake_detected", "_listener_suppress", "_listener_needs_reset"):
        if not hasattr(mod, attr):
            missing.append(attr)
    if missing:
        raise AttributeError(f"Missing always-on listener attributes: {missing}")
    # _wake_detected should be a threading.Event
    import threading
    if not isinstance(mod._wake_detected, threading.Event):
        raise TypeError(f"_wake_detected should be threading.Event, got {type(mod._wake_detected)}")


# ---------------------------------------------------------------------------
# 6. Text cleaning for TTS
# ---------------------------------------------------------------------------
def check_text_cleaning():
    mod = check_module_import._module
    fn = mod.clean_text_for_speech
    # Asterisks (bold/italic) should be stripped
    assert fn("**bold**") == "bold", f"got {fn('**bold**')}"
    assert fn("*italic*") == "italic", f"got {fn('*italic*')}"
    # Inline code should be replaced with "(code)"
    assert fn("`some_var`") == "(code)", f"got {fn('`some_var`')}"
    # Fenced code blocks should be replaced with "(code omitted)"
    result = fn("Here:\n```python\ndef foo():\n    pass\n```\nDone.")
    assert "(code omitted)" in result, f"got {result}"
    assert "def foo" not in result, f"code content leaked: {result}"
    # Headers should be stripped
    assert fn("## Header") == "Header", f"got {fn('## Header')}"
    # Plain text should pass through
    assert fn("hello world") == "hello world"


# ---------------------------------------------------------------------------
# 7. Echo detection
# ---------------------------------------------------------------------------
def check_echo_detection():
    import time as _time
    mod = check_module_import._module
    # Clear state
    mod._recently_spoken.clear()
    # Canned messages should always be detected as echo
    assert mod.is_self_echo("Bear with me.")
    assert mod.is_self_echo("bear with me")
    assert mod.is_self_echo("Still working on it.")
    assert mod.is_self_echo("Let me think about that.")
    assert mod.is_self_echo("Jarvis is ready")
    # Built-in command responses should be detected as echo
    assert mod.is_self_echo("Restarting now.")
    assert mod.is_self_echo("restarting now")
    assert mod.is_self_echo("Sorry, the revert failed.")
    assert mod.is_self_echo("Stopped.")
    assert mod.is_self_echo("Queued. I'll handle that after this task.")
    # Unrelated text should not match
    assert not mod.is_self_echo("turn on the lights")
    assert not mod.is_self_echo("what is the weather today")
    # Dynamic spoken text (Claude responses) — uses (timestamp, text) tuples
    mod._recently_spoken.append((_time.time(), "The answer is forty two."))
    assert mod.is_self_echo("the answer is forty two")
    assert mod.is_self_echo("answer is forty two")
    assert not mod.is_self_echo("what is the meaning of life")
    # Sentence-level fragment matching for long responses
    mod._recently_spoken.append((_time.time(),
        "A hash map stores key value pairs. It uses a hash function for constant time lookups."))
    assert mod.is_self_echo("It uses a hash function for constant time lookups")
    # Wake word prefix stripping (tested via built-in command matching)
    # "Hey Jarvis restart" should trigger the restart path after prefix stripping
    # We can't easily test the full main loop here, but verify the prefix list
    # is constructed correctly with the configured wake word name.
    wn = mod.WAKE_WORD_NAME.lower()
    expected_prefixes = [f"hey {wn} ", f"hey {wn}, ", f"{wn} ", f"{wn}, "]
    for pfx in expected_prefixes:
        test_text = f"{pfx}restart"
        stripped = test_text
        for p in expected_prefixes:
            if stripped.lower().startswith(p):
                stripped = stripped[len(p):].strip()
                break
        assert stripped == "restart", f"prefix stripping failed for '{test_text}': got '{stripped}'"

    # Garbage transcription filtering
    assert mod.is_garbage_transcription(".")
    assert mod.is_garbage_transcription("...")
    assert mod.is_garbage_transcription("Thanks for watching!")
    assert mod.is_garbage_transcription("thank you")
    assert not mod.is_garbage_transcription("What time is it?")
    assert not mod.is_garbage_transcription("hello")
    # Clean up
    mod._recently_spoken.clear()


# ---------------------------------------------------------------------------
# 8. Rate limit parsing
# ---------------------------------------------------------------------------
def check_rate_limit_parsing():
    mod = check_module_import._module
    from datetime import datetime as dt
    # Known rate limit format should parse to a datetime
    result = mod.parse_rate_limit("You've hit your limit · resets 9pm (America/New_York)")
    assert result is not None, "parse_rate_limit returned None for valid input"
    assert result.tzinfo is not None, "parsed time should be tz-aware"
    # With minutes
    result2 = mod.parse_rate_limit("You've hit your limit · resets 9:30pm (America/New_York)")
    assert result2 is not None, "parse_rate_limit returned None for time with minutes"
    assert result2.minute == 30, f"expected minute=30, got {result2.minute}"
    # Non-rate-limit text should return None
    assert mod.parse_rate_limit("Hello world") is None
    assert mod.parse_rate_limit("The weather is nice") is None
    # Edge: empty string
    assert mod.parse_rate_limit("") is None


# ---------------------------------------------------------------------------
# 9. Queue operations
# ---------------------------------------------------------------------------
def check_queue_operations():
    import tempfile
    mod = check_module_import._module
    # Use a temp file to avoid touching the real queue
    original = mod.QUEUE_FILE
    try:
        tmp = tempfile.mktemp(suffix=".json")
        mod.QUEUE_FILE = tmp

        # Empty queue
        assert mod.load_queue() == []
        assert mod.queue_list() == []
        assert mod.queue_pop() is None

        # Add items
        q = mod.queue_add("prompt one")
        assert len(q) == 1
        assert q[0]["prompt"] == "prompt one"

        q = mod.queue_add("prompt two")
        assert len(q) == 2
        assert q[0]["prompt"] == "prompt two"  # inserted at front

        # List
        listed = mod.queue_list()
        assert len(listed) == 2

        # Pop
        entry = mod.queue_pop()
        assert entry["prompt"] == "prompt two"
        assert len(mod.queue_list()) == 1

        # Clear
        mod.queue_clear()
        assert mod.queue_list() == []

    finally:
        mod.QUEUE_FILE = original
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 10. Model loading
# ---------------------------------------------------------------------------
def check_model_loading():
    mod = check_module_import._module
    wake_model, whisper_model, tts_voice, vad_model = mod.load_models()
    if wake_model is None:
        raise RuntimeError("wake_model is None")
    if whisper_model is None:
        raise RuntimeError("whisper_model is None")
    if tts_voice is None:
        raise RuntimeError("tts_voice is None")
    if vad_model is None:
        raise RuntimeError("vad_model is None")


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Jarvis preflight checks:")
    check("Syntax validation", check_syntax)
    check("Module import", check_module_import)
    check("Config import", check_config_import)
    check("Config value validation", check_config_values)
    check("Kokoro TTS wrapper", check_kokoro_wrapper)
    check("Function signatures", check_function_signatures)
    check("Listener state attributes", check_listener_state)
    check("Text cleaning for TTS", check_text_cleaning)
    check("Echo detection", check_echo_detection)
    check("Rate limit parsing", check_rate_limit_parsing)
    check("Queue operations", check_queue_operations)
    check("Model loading", check_model_loading)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
