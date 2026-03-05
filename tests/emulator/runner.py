"""Runner: patches jarvis module, runs main(), validates results."""

import importlib
import importlib.util
import io
import os
import subprocess
import sys
import threading

from .mocks import (
    JarvisExit,
    MockClaudeProcess,
    MockPulsePlayer,
    MockPulseRecorder,
    MockTTS,
    MockVAD,
    MockWakeModel,
    MockWhisper,
    mock_subprocess_run,
)
from .scenario import ScenarioDriver


JARVIS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_jarvis():
    """Import (or reimport) the jarvis module."""
    jarvis_path = os.path.join(JARVIS_DIR, "jarvis.py")
    spec = importlib.util.spec_from_file_location("jarvis", jarvis_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reset_module_state(jarvis):
    """Reset mutable module-level state between scenarios."""
    jarvis._recently_spoken.clear()
    jarvis._canned_normalized = None
    jarvis._last_echo_text = None
    jarvis._echo_repeat_count = 0
    # Reset send_to_claude's mutable default (first_call flag)
    sig = jarvis.send_to_claude.__code__
    for const in sig.co_consts:
        if isinstance(const, tuple):
            # Can't mutate tuples; the mutable default is a list
            pass
    # The mutable default `first_call=[True]` is stored in __defaults__
    defaults = jarvis.send_to_claude.__defaults__
    if defaults:
        for d in defaults:
            if isinstance(d, list) and len(d) == 1 and isinstance(d[0], bool):
                d[0] = True


class ScenarioReport:
    """Results from running a single scenario."""

    def __init__(self, name):
        self.name = name
        self.passed = True
        self.errors = []
        self.events = []
        self.exit_code = None

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"<ScenarioReport {self.name}: {status}>"


def run_scenario(scenario_path, time_scale=None, debug=False):
    """Run a single YAML scenario through the real jarvis.main().

    Returns a ScenarioReport with pass/fail status and recorded events.
    """
    driver = ScenarioDriver(scenario_path)
    if time_scale is not None:
        driver.time_scale = time_scale

    report = ScenarioReport(driver.name)

    # Import jarvis module
    try:
        jarvis = _import_jarvis()
    except Exception as e:
        report.fail(f"Failed to import jarvis: {e}")
        return report

    # Reset module state
    _reset_module_state(jarvis)

    # Enable debug flags if requested
    if debug:
        jarvis.DEBUG_MODELS = True
        jarvis.DEBUG_RECORDING = True
        jarvis.DEBUG_TRANSCRIPTION = True
        jarvis.DEBUG_CLAUDE = True
        jarvis.DEBUG_TTS = True
        jarvis.DEBUG_ECHO = True

    # Create mock instances
    mock_wake = MockWakeModel(driver)
    mock_vad = MockVAD(driver)
    mock_whisper = MockWhisper(driver)
    mock_tts = MockTTS(driver)

    # Patch transcribe to return scripted text (no temp WAV file needed).
    # Interaction advancement happens when the NEXT wake word fires in
    # MockWakeModel.predict(), so all events during an interaction cycle
    # (transcription, echo check, Claude call, TTS) are tagged with the
    # correct interaction index.
    original_transcribe = jarvis.transcribe

    def patched_transcribe(whisper_model, audio_bytes):
        """Return scripted transcription from current interaction."""
        interaction = driver.current_interaction()
        text = ""
        if interaction:
            text = interaction.get("transcription", "")
        driver.record_event("transcription", {"text": text})
        return text

    # Patch PulseRecorder and PulsePlayer
    original_PulseRecorder = jarvis.PulseRecorder
    original_PulsePlayer = jarvis.PulsePlayer
    original_load_models = jarvis.load_models
    original_os_exit = os._exit
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def mock_pulse_recorder(rate=16000, channels=1, chunk=1280):
        return MockPulseRecorder(driver, rate, channels, chunk)

    def mock_pulse_player(rate=22050, channels=1):
        return MockPulsePlayer(driver, rate, channels)

    def mock_load_models():
        return mock_wake, mock_whisper, mock_tts, mock_vad

    def mock_os_exit(code):
        raise JarvisExit(code)

    def mock_popen(cmd, **kwargs):
        return MockClaudeProcess(driver, cmd, **kwargs)

    def mock_run_fn(args, **kwargs):
        return mock_subprocess_run(driver, args, **kwargs)

    # Apply patches
    jarvis.PulseRecorder = mock_pulse_recorder
    jarvis.PulsePlayer = mock_pulse_player
    jarvis.load_models = mock_load_models
    jarvis.transcribe = patched_transcribe
    os._exit = mock_os_exit
    subprocess.Popen = mock_popen
    subprocess.run = mock_run_fn

    # Suppress stdout during scenario run
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = captured_stdout
        jarvis.main()
        report.exit_code = 0
    except JarvisExit as e:
        report.exit_code = e.code
    except Exception as e:
        report.fail(f"Unexpected exception: {type(e).__name__}: {e}")
    finally:
        sys.stdout = old_stdout
        # Restore patches
        jarvis.PulseRecorder = original_PulseRecorder
        jarvis.PulsePlayer = original_PulsePlayer
        jarvis.load_models = original_load_models
        jarvis.transcribe = original_transcribe
        os._exit = original_os_exit
        subprocess.Popen = original_popen
        subprocess.run = original_run

    report.events = driver.events

    # Validate expectations
    _validate(driver, report)

    return report


def _validate(driver, report):
    """Check scenario expectations against recorded events."""
    for i, interaction in enumerate(driver.interactions):
        expected = interaction.get("expected", {})
        if not expected:
            continue

        events = driver.events_for_interaction(i)
        tts_events = [e for e in events if e["type"] == "tts"]
        claude_events = [e for e in events if e["type"] == "claude_called"]

        # Check spoken_response
        if "spoken_response" in expected:
            expected_text = expected["spoken_response"]
            spoken_texts = [e["data"]["text"] for e in tts_events]
            if expected_text and expected_text not in spoken_texts:
                report.fail(
                    f"Interaction {i}: expected spoken_response '{expected_text}', "
                    f"got TTS events: {spoken_texts}"
                )
            elif not expected_text and spoken_texts:
                # Expected no speech but got some (excluding fillers)
                pass  # Fillers are acceptable

        # Check echo_filtered
        if "echo_filtered" in expected:
            should_filter = expected["echo_filtered"]
            was_called = len(claude_events) > 0
            if should_filter and was_called:
                report.fail(
                    f"Interaction {i}: expected echo to be filtered, but Claude was called"
                )
            elif not should_filter and not was_called:
                # Check if this interaction even has a transcription that should reach Claude
                text = interaction.get("transcription", "")
                if text and not _is_builtin_command(text):
                    report.fail(
                        f"Interaction {i}: expected Claude to be called, but it wasn't"
                    )

        # Check claude_called
        if "claude_called" in expected:
            should_call = expected["claude_called"]
            was_called = len(claude_events) > 0
            if should_call != was_called:
                report.fail(
                    f"Interaction {i}: expected claude_called={should_call}, "
                    f"got {was_called}"
                )

        # Check exit_code
        if "exit_code" in expected:
            if report.exit_code != expected["exit_code"]:
                report.fail(
                    f"Interaction {i}: expected exit_code={expected['exit_code']}, "
                    f"got {report.exit_code}"
                )

        # Check min_filler_count
        if "min_filler_count" in expected:
            min_count = expected["min_filler_count"]
            # Count TTS events that aren't the main response
            main_response = interaction.get("claude", {}).get("response", "")
            filler_count = sum(
                1 for e in tts_events
                if e["data"]["text"] != main_response
            )
            if filler_count < min_count:
                report.fail(
                    f"Interaction {i}: expected at least {min_count} filler messages, "
                    f"got {filler_count}"
                )


def _is_builtin_command(text):
    """Check if text matches a built-in command."""
    lower = text.lower().strip().rstrip(".")
    builtin_phrases = (
        "restart", "restart yourself", "reboot", "reboot yourself",
        "revert", "revert yourself", "revert the last change",
        "undo the last change", "roll back", "rollback",
        "please restart",
    )
    return lower in builtin_phrases
