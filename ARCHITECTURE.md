# Jarvis Architecture

Voice assistant that listens for a wake word, records speech, transcribes it, sends it to Claude Code, and speaks the response aloud.

## Project Structure

- `jarvis.py` — Main application code.
- `config/config.py` — All tunable constants and spoken message lists.
- `agents/main/CLAUDE.md` — System prompt for Claude Code sessions.
- `logs/` — Runtime logs (gitignored): `error.log`, `conversation.log`.
- `voices/` — Piper TTS voice models (downloaded on first run).
- `test_preflight.py` — Pre-restart validation checks (safety gate).
- `jarvis.sh` — Launcher script with auto-restart on exit code 42 and preflight gate.
- `jarvis.service` — systemd unit file.

## Components

### Audio I/O (`PulseRecorder`, `PulsePlayer`)
Direct PulseAudio access via ctypes (libpulse-simple). No PyAudio dependency.
- **PulseRecorder**: Captures 16kHz mono int16 audio in configurable chunk sizes.
- **PulsePlayer**: Streams int16 audio for TTS playback with drain/flush support.

### Wake Word Detection (openwakeword)
Uses the `hey_jarvis_v0.1.onnx` model from openwakeword. Runs on ONNX runtime.
- Feeds 1280-sample (80ms) chunks continuously.
- Activation threshold: configurable (`WAKE_WORD_THRESHOLD`, default 0.8).

### Voice Activity Detection (Silero VAD)
ML-based speech/silence detection using `silero-vad` (ONNX mode).
- Processes 512-sample chunks at 16kHz (32ms each).
- Speech probability > 0.5 = speech detected.
- End-of-speech: rolling window of 1.0s must be 80% silent.
- Safety cap: 15 seconds max recording.

### Speech-to-Text (faster-whisper)
Uses `small.en` model with int8 quantization on CPU.
- Writes audio to a temp WAV file, transcribes, deletes.

### Text-to-Speech (Piper)
Uses `en_US-lessac-medium` voice model.
- Streams synthesized audio chunks to PulsePlayer.
- Playback is blocking — microphone is not monitored during TTS to prevent self-triggering (the mic picking up the speaker output and detecting a false wake word).

### Claude Code Integration
Sends transcribed text to `claude` CLI in a subprocess.
- Uses session IDs for conversation continuity (`--session-id` / `--resume`).
- Configurable timeout (`CLAUDE_TIMEOUT`, default 300s).
- Spoken acknowledgements if response takes longer than `INITIAL_ACK_DELAY` (default 10s).

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
6. Speak response (blocking).
7. Reset wake word model state, loop back.

## Configuration
All tunable constants live in `config/config.py`. Key values:

| Constant | Value | Purpose |
|---|---|---|
| RATE | 16000 | Audio sample rate (Hz) |
| CHUNK | 1280 | Wake word detection chunk size (80ms) |
| VAD_CHUNK | 512 | Silero VAD chunk size (32ms) |
| VAD_THRESHOLD | 0.5 | Speech probability threshold |
| WAKE_WORD_THRESHOLD | 0.9 | Wake word activation threshold |
| SILENCE_DURATION | 1.0s | Silence window to end recording |
| MAX_RECORD_SECONDS | 15 | Recording safety cap |
| CLAUDE_TIMEOUT | 300 | Claude API timeout (seconds) |
| INITIAL_ACK_DELAY | 10.0 | Seconds before first spoken filler |

## Dependencies
- **openwakeword**: Wake word detection (ONNX)
- **silero-vad**: Voice activity detection (ONNX)
- **faster-whisper**: Speech-to-text (CTranslate2)
- **piper-tts**: Text-to-speech (ONNX)
- **numpy**: Audio array processing
- **torch**: Tensor ops for Silero VAD

## Preflight Validation (`test_preflight.py`)

Safety gate between code changes and restart. Runs automatically in `jarvis.sh` after exit code 42, before spawning a new Jarvis process.

### Checks (ordered fastest-first)
1. **Syntax validation** — `py_compile` on `jarvis.py`, `config/config.py`, `config/__init__.py`.
2. **Config import** — Mirrors the exact import statement from `jarvis.py`.
3. **Config value validation** — Type checks, range checks (e.g., `0 < VAD_THRESHOLD <= 1.0`), non-empty message lists.
4. **Module import** — `importlib` loads `jarvis.py` (catches broken imports, missing deps).
5. **Function signatures** — Verifies `load_models`, `record_until_silence`, `transcribe`, `speak`, `send_to_claude`, `main` exist with expected params.
6. **Model loading** — Loads all 4 models (wake word, whisper, VAD, TTS).

Exits 0 on pass, 1 on fail. All checks run (no early exit) to give a complete picture.

### Auto-revert flow (in `jarvis.sh`)
1. Jarvis exits with code 42 (restart requested).
2. `test_preflight.py` is syntax-checked first, then run with a 120s timeout.
3. If preflight passes → restart normally.
4. If preflight fails → `git revert --no-edit HEAD`.
   - If revert succeeds, re-run preflight and restart.
   - If revert fails, fallback: `git checkout HEAD~1 -- jarvis.py config/config.py`.

## Process Management
Runs as a systemd service. Exit code 42 triggers restart (used by built-in restart/revert commands and SIGUSR1). SIGINT triggers graceful shutdown with spoken goodbye.
