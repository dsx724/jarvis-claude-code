"""Mock classes that replace hardware and external dependencies for emulation testing."""

import json
import threading
import time


class JarvisExit(Exception):
    """Raised when os._exit is called during emulation."""
    def __init__(self, code):
        self.code = code
        super().__init__(f"os._exit({code})")


class MockPulseRecorder:
    """Replaces PulseRecorder. Returns zero bytes; raises JarvisExit when scenario exhausted."""

    def __init__(self, driver, rate=16000, channels=1, chunk=1280):
        self.driver = driver
        self.chunk = chunk
        self.frame_size = 2 * channels

    def read(self, n_frames, exception_on_overflow=False):
        if self.driver.is_exhausted():
            # On background threads (listen_for_wake_word), don't raise — just
            # block until the thread is interrupted via interrupt_event/join.
            if threading.current_thread() is not threading.main_thread():
                # Sleep a long time; the daemon thread will be killed when main exits
                time.sleep(100)
                return b'\x00' * (n_frames * self.frame_size)
            time.sleep(0.01 * self.driver.time_scale)
            raise JarvisExit(0)
        time.sleep(0.001 * self.driver.time_scale)
        return b'\x00' * (n_frames * self.frame_size)

    def flush(self):
        pass

    def close(self):
        pass


class MockPulsePlayer:
    """Replaces PulsePlayer. No-op writes; records audio events."""

    def __init__(self, driver, rate=22050, channels=1):
        self.driver = driver

    def write(self, data):
        pass

    def drain(self):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class MockPreprocessor:
    """Minimal preprocessor stub for openwakeword model compatibility."""

    def __init__(self):
        self.raw_data_buffer = []
        self.melspectrogram_buffer = None
        self.accumulated_samples = 0
        self.feature_buffer = None

    def clear(self):
        self.raw_data_buffer = []

    def _get_embeddings(self, data):
        return None


class MockWakeModel:
    """Replaces openwakeword Model. Returns low scores until trigger point."""

    def __init__(self, driver):
        self.driver = driver
        self.preprocessor = MockPreprocessor()
        self._chunk_count = 0
        self._triggered = False
        self._trigger_count = 0

    def predict(self, audio_data):
        interaction = self.driver.current_interaction()
        if interaction is None:
            return {"hey_jarvis_v0.1": 0.0}

        wake_cfg = interaction.get("wake_word", {})
        silent_chunks = wake_cfg.get("silent_chunks", 5)
        trigger_score = wake_cfg.get("trigger_score", 0.95)

        # Distinguish main loop vs interrupt listener thread
        is_interrupt_thread = threading.current_thread() is not threading.main_thread()

        if is_interrupt_thread:
            # For interrupt listener: trigger if interaction says so
            if interaction.get("_interrupt_wake", False):
                return {"hey_jarvis_v0.1": trigger_score}
            return {"hey_jarvis_v0.1": 0.0}

        # Main loop wake word detection
        if not self._triggered:
            self._chunk_count += 1
            if self._chunk_count > silent_chunks:
                self._triggered = True
                self._chunk_count = 0
                # Advance to next interaction when the 2nd+ wake word fires.
                # This ensures all events from the previous cycle (Claude, TTS)
                # are tagged with the correct interaction index.
                if self._trigger_count > 0:
                    self.driver.advance()
                self._trigger_count += 1
                return {"hey_jarvis_v0.1": trigger_score}
        return {"hey_jarvis_v0.1": 0.0}

    def reset(self):
        self._triggered = False
        self._chunk_count = 0


class MockVAD:
    """Replaces Silero VAD. High prob for N speech_chunks, then low."""

    def __init__(self, driver):
        self.driver = driver
        self._speech_chunk_count = 0

    def __call__(self, audio_tensor, rate):
        interaction = self.driver.current_interaction()
        if interaction is None:
            return _MockTensor(0.0)

        rec_cfg = interaction.get("recording", {})
        speech_chunks = rec_cfg.get("speech_chunks", 20)

        self._speech_chunk_count += 1
        if self._speech_chunk_count <= speech_chunks:
            return _MockTensor(0.9)  # Above VAD threshold
        return _MockTensor(0.1)  # Below VAD threshold (silence)

    def reset_states(self):
        self._speech_chunk_count = 0


class _MockTensor:
    """Minimal tensor-like object with .item() method."""
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class MockWhisper:
    """Replaces WhisperModel. Returns scripted transcription."""

    def __init__(self, driver):
        self.driver = driver

    def transcribe(self, path, **kwargs):
        interaction = self.driver.current_interaction()
        text = ""
        if interaction:
            text = interaction.get("transcription", "")
        segments = [_MockSegment(text)] if text else []
        return segments, None


class _MockSegment:
    """Minimal whisper segment with .text attribute."""
    def __init__(self, text):
        self.text = text


class MockTTS:
    """Replaces PiperVoice. Records synthesized text, yields minimal audio."""

    def __init__(self, driver):
        self.driver = driver
        self.config = _MockTTSConfig()

    def synthesize(self, text):
        self.driver.record_event("tts", {"text": text})
        yield _MockAudioChunk()

    @staticmethod
    def load(path):
        # Not used in emulation — the runner creates MockTTS directly
        raise NotImplementedError("MockTTS.load should not be called")


class _MockTTSConfig:
    """Minimal config with sample_rate."""
    def __init__(self):
        self.sample_rate = 22050


class _MockAudioChunk:
    """Minimal audio chunk from TTS synthesis."""
    def __init__(self):
        import numpy as np
        self.audio_float_array = np.zeros(1024, dtype=np.float32)


class MockClaudeProcess:
    """Replaces subprocess.Popen for Claude CLI calls.

    Streams scripted JSON events on stdout with optional delay.
    """

    def __init__(self, driver, cmd, **kwargs):
        self.driver = driver
        self.returncode = 0
        self._cmd = cmd

        interaction = driver.current_interaction()
        claude_cfg = interaction.get("claude", {}) if interaction else {}

        delay = claude_cfg.get("delay", 0.1) * driver.time_scale
        response_text = claude_cfg.get("response", "")
        tool_events = claude_cfg.get("tool_events", [])

        # Build the event stream
        lines = []
        for tool_evt in tool_events:
            evt = {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": tool_evt.get("name", "Bash"),
                         "input": tool_evt.get("input", {})}
                    ]
                }
            }
            lines.append(json.dumps(evt))

        result_evt = {"type": "result", "result": response_text}
        lines.append(json.dumps(result_evt))

        self._output = "\n".join(lines) + "\n"
        self._delay = delay
        self._stdout_lines = None
        self._stderr_text = ""

        self.stdout = _DelayedLineIterator(self._output, self._delay)
        self.stderr = _MockStderr(self._stderr_text)

        driver.record_event("claude_called", {
            "prompt": cmd[-1] if cmd else "",
            "response": response_text,
        })

    def kill(self):
        pass

    def wait(self):
        self.returncode = 0
        return 0


class _DelayedLineIterator:
    """Iterates over lines with an initial delay to simulate Claude processing time."""

    def __init__(self, text, delay):
        self._lines = text.splitlines(keepends=True)
        self._delay = delay
        self._first = True

    def __iter__(self):
        for line in self._lines:
            if self._first:
                self._first = False
                time.sleep(self._delay)
            yield line

    def read(self):
        return "".join(self._lines)


class _MockStderr:
    def __init__(self, text=""):
        self._text = text

    def read(self):
        return self._text

    def strip(self):
        return self._text.strip()


def mock_subprocess_run(driver, args, **kwargs):
    """Handles git rev-parse/revert for built-in commands."""
    if not isinstance(args, (list, tuple)):
        args = [args]

    cmd = " ".join(str(a) for a in args)

    if "git" in cmd and "rev-parse" in cmd:
        return _MockCompletedProcess(stdout="abc1234\n", returncode=0)

    if "git" in cmd and "revert" in cmd:
        driver.record_event("git_revert", {"cmd": cmd})
        return _MockCompletedProcess(stdout="", returncode=0)

    # Default: success
    return _MockCompletedProcess(stdout="", returncode=0)


class _MockCompletedProcess:
    """Minimal subprocess.CompletedProcess replacement."""
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
