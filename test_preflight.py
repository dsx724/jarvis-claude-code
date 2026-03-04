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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    files = ["jarvis.py", "config/config.py", "config/__init__.py"]
    for f in files:
        path = os.path.join(SCRIPT_DIR, f)
        py_compile.compile(path, doraise=True)


# ---------------------------------------------------------------------------
# 2. Config import — mirrors jarvis.py lines 25-31
# ---------------------------------------------------------------------------
def check_config_import():
    sys.path.insert(0, SCRIPT_DIR)
    from config import (
        RATE, CHANNELS, CHUNK,
        VAD_THRESHOLD, VAD_CHUNK, SILENCE_DURATION, SILENCE_RATIO, MAX_RECORD_SECONDS,
        WAKE_WORD_THRESHOLD, CLAUDE_TIMEOUT, INITIAL_ACK_DELAY,
        ACKNOWLEDGEMENTS, STILL_WORKING, STILL_WORKING_INTERVAL,
        STARTUP_MESSAGES, SHUTDOWN_MESSAGES,
    )


# ---------------------------------------------------------------------------
# 3. Config value validation
# ---------------------------------------------------------------------------
def check_config_values():
    sys.path.insert(0, SCRIPT_DIR)
    from config import (
        RATE, CHANNELS, CHUNK,
        VAD_THRESHOLD, VAD_CHUNK, SILENCE_DURATION, SILENCE_RATIO, MAX_RECORD_SECONDS,
        WAKE_WORD_THRESHOLD, CLAUDE_TIMEOUT, INITIAL_ACK_DELAY,
        ACKNOWLEDGEMENTS, STILL_WORKING, STILL_WORKING_INTERVAL,
        STARTUP_MESSAGES, SHUTDOWN_MESSAGES,
    )
    errors = []

    # Type checks
    for name, val, expected in [
        ("RATE", RATE, int),
        ("CHANNELS", CHANNELS, int),
        ("CHUNK", CHUNK, int),
        ("VAD_CHUNK", VAD_CHUNK, int),
        ("MAX_RECORD_SECONDS", MAX_RECORD_SECONDS, (int, float)),
        ("CLAUDE_TIMEOUT", CLAUDE_TIMEOUT, (int, float)),
        ("STILL_WORKING_INTERVAL", STILL_WORKING_INTERVAL, (int, float)),
        ("INITIAL_ACK_DELAY", INITIAL_ACK_DELAY, (int, float)),
        ("VAD_THRESHOLD", VAD_THRESHOLD, (int, float)),
        ("SILENCE_DURATION", SILENCE_DURATION, (int, float)),
        ("SILENCE_RATIO", SILENCE_RATIO, (int, float)),
        ("WAKE_WORD_THRESHOLD", WAKE_WORD_THRESHOLD, (int, float)),
    ]:
        if not isinstance(val, expected):
            errors.append(f"{name} should be {expected}, got {type(val)}")

    # Range checks
    if not (0 < VAD_THRESHOLD <= 1.0):
        errors.append(f"VAD_THRESHOLD={VAD_THRESHOLD} not in (0, 1.0]")
    if not (0 < WAKE_WORD_THRESHOLD <= 1.0):
        errors.append(f"WAKE_WORD_THRESHOLD={WAKE_WORD_THRESHOLD} not in (0, 1.0]")
    if not (0 < SILENCE_RATIO <= 1.0):
        errors.append(f"SILENCE_RATIO={SILENCE_RATIO} not in (0, 1.0]")
    if RATE <= 0:
        errors.append(f"RATE={RATE} must be positive")
    if CHUNK <= 0:
        errors.append(f"CHUNK={CHUNK} must be positive")
    if VAD_CHUNK <= 0:
        errors.append(f"VAD_CHUNK={VAD_CHUNK} must be positive")
    if CLAUDE_TIMEOUT <= 0:
        errors.append(f"CLAUDE_TIMEOUT={CLAUDE_TIMEOUT} must be positive")

    # Non-empty message lists
    for name, lst in [
        ("ACKNOWLEDGEMENTS", ACKNOWLEDGEMENTS),
        ("STILL_WORKING", STILL_WORKING),
        ("STARTUP_MESSAGES", STARTUP_MESSAGES),
        ("SHUTDOWN_MESSAGES", SHUTDOWN_MESSAGES),
    ]:
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
def check_function_signatures():
    mod = check_module_import._module
    expected = {
        "load_models": [],
        "record_until_silence": ["stream", "vad_model"],
        "transcribe": ["whisper_model", "audio_bytes"],
        "clean_text_for_speech": ["text"],
        "is_self_echo": ["transcribed"],
        "speak": ["tts_voice", "text"],
        "listen_for_wake_word": ["wake_model", "interrupt_event"],
        "send_to_claude": ["text"],
        "main": [],
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
# 6. Text cleaning for TTS
# ---------------------------------------------------------------------------
def check_text_cleaning():
    mod = check_module_import._module
    fn = mod.clean_text_for_speech
    # Asterisks (bold/italic) should be stripped
    assert fn("**bold**") == "bold", f"got {fn('**bold**')}"
    assert fn("*italic*") == "italic", f"got {fn('*italic*')}"
    # Backticks should be stripped
    assert fn("`code`") == "code", f"got {fn('`code`')}"
    # Headers should be stripped
    assert fn("## Header") == "Header", f"got {fn('## Header')}"
    # Plain text should pass through
    assert fn("hello world") == "hello world"


# ---------------------------------------------------------------------------
# 7. Echo detection
# ---------------------------------------------------------------------------
def check_echo_detection():
    mod = check_module_import._module
    # Clear state
    mod._recently_spoken.clear()
    # No spoken text — nothing should be an echo
    assert not mod.is_self_echo("hello world")
    # Add spoken text and check exact/fuzzy match
    mod._recently_spoken.append("Let me think about that.")
    assert mod.is_self_echo("let me think about that")
    assert mod.is_self_echo("Let me think about that.")
    # Partial match (substring)
    assert mod.is_self_echo("think about that")
    # Unrelated text should not match
    assert not mod.is_self_echo("turn on the lights")
    # Clean up
    mod._recently_spoken.clear()


# ---------------------------------------------------------------------------
# 8. Model loading
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
    check("Config import", check_config_import)
    check("Config value validation", check_config_values)
    check("Module import", check_module_import)
    check("Function signatures", check_function_signatures)
    check("Text cleaning for TTS", check_text_cleaning)
    check("Echo detection", check_echo_detection)
    check("Model loading", check_model_loading)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
