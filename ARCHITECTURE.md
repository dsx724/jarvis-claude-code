# Jarvis Architecture

Voice assistant that listens for a wake word, records speech, transcribes it, sends it to Claude Code, and speaks the response aloud.

## Components

### Audio I/O (`PulseRecorder`, `PulsePlayer`)
Direct PulseAudio access via ctypes (libpulse-simple). No PyAudio dependency.
- **PulseRecorder**: Captures 16kHz mono int16 audio in configurable chunk sizes.
- **PulsePlayer**: Streams int16 audio for TTS playback with drain/flush support.

### Wake Word Detection (openwakeword)
Uses the `hey_jarvis_v0.1.onnx` model from openwakeword. Runs on ONNX runtime.
- Feeds 1280-sample (80ms) chunks continuously.
- Activation threshold: 0.8.

### Voice Activity Detection (Silero VAD)
ML-based speech/silence detection using `silero-vad` (ONNX mode).
- Processes 512-sample chunks at 16kHz (32ms each).
- Speech probability > 0.5 = speech detected.
- End-of-speech: rolling window of 1.0s must be 80% silent.
- Safety cap: 15 seconds max recording.
- Replaced previous RMS-based `NoiseTracker` approach which was unreliable with background noise.

### Speech-to-Text (faster-whisper)
Uses `small.en` model with int8 quantization on CPU.
- Writes audio to a temp WAV file, transcribes, deletes.

### Text-to-Speech (Piper)
Uses `en_US-lessac-medium` voice model.
- Streams synthesized audio chunks to PulsePlayer.
- Supports interruptible playback via `stop_event`.

### Claude Code Integration
Sends transcribed text to `claude` CLI in a subprocess.
- Uses session IDs for conversation continuity (`--session-id` / `--resume`).
- 300-second timeout per request.
- Spoken acknowledgements ("Let me think...") if response takes >10 seconds.

### Built-in Commands
Handled locally without calling Claude:
- **restart** / **reboot**: Exits with code 42 (systemd restarts).
- **revert**: Runs `git revert HEAD` and restarts.

## Main Loop Flow
1. Read audio chunk → feed to wake word model.
2. On activation → `record_until_silence()` with Silero VAD.
3. Transcribe with faster-whisper.
4. Check for built-in commands.
5. Send to Claude Code (with timeout-based spoken fillers).
6. Speak response in background thread while monitoring for wake word interrupt.
7. If interrupted mid-speech, record new utterance and loop back.

## Key Constants
| Constant | Value | Purpose |
|---|---|---|
| RATE | 16000 | Audio sample rate (Hz) |
| CHUNK | 1280 | Wake word detection chunk size (80ms) |
| VAD_CHUNK | 512 | Silero VAD chunk size (32ms) |
| VAD_THRESHOLD | 0.5 | Speech probability threshold |
| SILENCE_DURATION | 1.0s | Silence window to end recording |
| MAX_RECORD_SECONDS | 15 | Recording safety cap |

## Dependencies
- **openwakeword**: Wake word detection (ONNX)
- **silero-vad**: Voice activity detection (ONNX)
- **faster-whisper**: Speech-to-text (CTranslate2)
- **piper-tts**: Text-to-speech (ONNX)
- **numpy**: Audio array processing
- **torch**: Tensor ops for Silero VAD

## Process Management
Runs as a systemd service. Exit code 42 triggers restart (used by built-in restart/revert commands and SIGUSR1). SIGINT triggers graceful shutdown with spoken goodbye.
