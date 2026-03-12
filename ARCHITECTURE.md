[jarvis](.) / [ARCHITECTURE.md](ARCHITECTURE.md)

# Jarvis Architecture

Voice assistant that listens for a wake word, records speech, transcribes it, sends it to Claude Code, and speaks the response aloud.

## Project Structure

- [jarvis.py](jarvis.py) -- Main application code (loads config directly from [config/jarvis.ini](config/jarvis.ini)).
- [config/jarvis.ini](config/jarvis.ini) -- User-editable configuration file (INI format) with all tunable settings.
- [agents/main/CLAUDE.md](agents/main/CLAUDE.md) -- System prompt for Claude Code sessions.
- [logs/](logs/) -- Runtime logs (gitignored): `error.log`, `conversation.log`.
- [voices/](voices/) -- TTS voice models (Piper auto-downloaded on first run; Kokoro requires manual download).
- [tests/test_preflight.py](tests/test_preflight.py) -- Pre-restart validation checks (safety gate).
- [tests/test_emulator.py](tests/test_emulator.py) -- Emulation testing CLI (runs YAML scenarios through real Jarvis logic).
- [tests/emulator/](tests/emulator/) -- Mock classes, scenario driver, and runner for emulation tests.
- [tests/scenarios/](tests/scenarios/) -- YAML scenario definitions for emulation testing.
- [jarvis.sh](jarvis.sh) -- Launcher script with auto-restart on exit code 42 and preflight gate.
- [jarvis.service](jarvis.service) -- systemd unit file.

## Components

### Audio I/O (`PulseRecorder`, `PulsePlayer`)
Direct PulseAudio access via ctypes (libpulse-simple). No PyAudio dependency.
- **PulseRecorder**: Captures 16kHz mono int16 audio in configurable chunk sizes.
- **PulsePlayer**: Streams int16 audio for TTS playback with drain/flush support.

### Wake Word Detection (openwakeword)
Uses openwakeword with a configurable wake word (`WAKE_WORD_NAME`, default `jarvis`). Model file is resolved as `hey_<name>_v0.1.onnx`. Supported names: `jarvis`, `marvin`, `mycroft`. Runs on ONNX runtime.
- Feeds 1280-sample (80ms) chunks continuously.
- Activation threshold: configurable (`WAKE_WORD_THRESHOLD`, default 0.9).
- The wake word name is used throughout: model selection, TTS filtering (prevents TTS from saying the wake word and re-triggering), built-in command phrases (e.g. "restart jarvis"), and the ready message.

#### Always-On Background Listener
A persistent daemon thread (`_always_on_listener`) runs for the entire process lifetime with its own `bg_wake_model` instance (separate from the main thread's `wake_model` to avoid corrupting `predict()`'s internal sliding window state). Sets `_wake_detected` event when the wake word is heard.

**Suppression timeline:**
| Phase | Background listener | Main thread |
|---|---|---|
| IDLE | ACTIVE | Reads chunks + predicts |
| RECORDING | SUPPRESSED (`_listener_suppress`) | Records audio |
| CLAUDE WAIT | ACTIVE | Checks `_wake_detected` in wait timeouts |
| TTS | ACTIVE | `speak()` checks `_wake_detected` in sub-chunk loop |
| POST-TTS | Resets via `_listener_needs_reset` flag | Flushes stream, resets main wake model |

This enables:
1. **"Stop" builtin**: User can say "stop" or "cancel" during Claude processing to kill the subprocess.
2. **Prompt queuing**: User can queue new prompts while Claude is busy; they are processed after the current task.
3. **Unified TTS interruption**: No more per-speech listener threads; `speak_and_clear` passes `_wake_detected` as the interrupt event.

### Voice Activity Detection (Silero VAD)
ML-based speech/silence detection using `silero-vad` (ONNX mode).
- Processes 512-sample chunks at 16kHz (32ms each).
- Speech probability > 0.5 = speech detected.
- End-of-speech: rolling window of 1.0s must be 80% silent.
- Safety cap: 15 seconds max recording.

### Speech-to-Text (faster-whisper)
Uses configurable model (default `medium.en`) with int8 quantization on CPU. Set via `[stt] model` in `jarvis.ini`.
- Writes audio to a temp WAV file, transcribes, deletes.
- **Echo filtering**: After transcription, checks if the text matches something Jarvis recently said (via `is_self_echo()`). Uses fuzzy matching (`SequenceMatcher`) and substring checks against the last 3 spoken texts to filter out the mic picking up Jarvis's own TTS output.

### Text-to-Speech (Piper / Kokoro)
Configurable voice model via `jarvis.ini` (`[tts]` section). Supports two engines:
- **Piper** (default): Lightweight, fast. Voices auto-downloaded from Hugging Face. Default voice: `en_US-lessac-medium`. Sample rate: 22050Hz.
- **Kokoro** (`kokoro-onnx`): Higher quality voices. Requires `kokoro-v1.0.onnx` and `voices-v1.0.bin` in `voices/` (download from GitHub releases). Default voice: `af_heart`. Sample rate: 24000Hz. Wrapped by `KokoroTTS` class for Piper-compatible interface.
- Streams synthesized audio chunks to PulsePlayer.
- Playback is interruptible by the always-on background listener via `_wake_detected` event.
- Markdown formatting (asterisks, backticks, headers, bullets) is stripped before synthesis via `clean_text_for_speech()`.

### Claude Code Integration
Sends transcribed text to `claude` CLI via `subprocess.Popen` with `--output-format stream-json`.
- Uses session IDs for conversation continuity (`--session-id` / `--resume`).
- Configurable timeout (`CLAUDE_TIMEOUT`, default 300s) via `threading.Timer`.
- Streams JSON events from stdout; parses `tool_use` events to push live status messages (e.g. "Reading jarvis.py", "Running a command") to a `Queue`.
- Spoken acknowledgements if response takes longer than `INITIAL_ACK_DELAY` (default 10s). Subsequent filler messages prefer the latest tool status from the queue; falls back to canned `STILL_WORKING` messages when no tool events are available.

### Built-in Commands
Handled locally without calling Claude:
- **shutdown** / **shut down** / **go to sleep** / **goodnight**: Exits with code 0 (clean stop, no restart).
- **restart** / **reboot**: Exits with code 42 (systemd restarts).
- **revert**: Runs `git revert HEAD` and restarts.
- **stop** / **cancel** / **never mind** / **nevermind**: Kills Claude mid-processing (only active during Claude wait loop).
- **queue** / **show queue** / **list queue** / **pending prompts**: Lists queued prompts.
- **clear queue** / **empty queue**: Clears the queue and cancels any pending retry timer.

### Persistent Prompt Queue
When Claude Code returns a rate limit message, the prompt is saved to `logs/prompt_queue.json` and a timer is set to auto-retry when the limit resets.

- **Queue file**: `logs/prompt_queue.json` — JSON list of `{"prompt": "...", "queued_at": "..."}` objects.
- **Rate limit detection**: `parse_rate_limit()` extracts the reset time from messages matching "hit your limit...resets Xpm (timezone)".
- **Timer/scheduling**: `schedule_queue_processing()` sets a `threading.Timer` that fires `_queue_ready_event` at the reset time. The main wake word idle loop checks this event and calls `_process_queue()` when ready.
- **Startup recovery**: On startup, if the queue file is non-empty, Jarvis announces the count and immediately attempts to process them (if still rate-limited, it re-queues with a new timer).
- **Queue operations**: `load_queue()`, `save_queue()`, `queue_add()`, `queue_pop()`, `queue_list()`, `queue_clear()` — all disk-backed, no in-memory cache.

## Main Loop Flow
1. Read audio chunk → feed to wake word model (also check `_wake_detected` from bg listener).
1b. If `_queue_ready_event` is set (rate limit expired), process queued prompts.
2. On activation → suppress bg listener, `record_until_silence()` with Silero VAD, unsuppress.
3. Transcribe with faster-whisper, strip wake word prefix via `_strip_wake_prefix()`.
4. Filter self-echo (discard if transcription matches recently spoken text).
5. Check for built-in commands (restart, revert, stop, queue, clear queue).
6. Send to Claude Code (with timeout-based spoken fillers); `proc_holder` tracks subprocess.
6a. During Claude wait: if `_wake_detected` fires, record + transcribe the interrupt.
   - "stop"/"cancel"/"never mind" → kill Claude process, speak "Stopped.", continue loop.
   - Other text → `queue_add()`, speak "Queued.", continue waiting for Claude.
6b. If response is a rate limit → queue prompt, schedule retry timer, skip speech.
7. Speak response (interruptible via `_wake_detected`).
8. Reset wake word model state (main + bg via `_listener_needs_reset`), loop back.

## Configuration
All tunable settings live in `config/jarvis.ini` (INI format), loaded directly by `jarvis.py` at startup. Key values:

| Constant | Value | Purpose |
|---|---|---|
| DEBUG_MODELS | false | Debug model loading times |
| DEBUG_RECORDING | false | Debug recording, VAD, and iteration timing |
| DEBUG_TRANSCRIPTION | false | Debug whisper transcription breakdown |
| DEBUG_CLAUDE | false | Debug Claude subprocess lifecycle |
| DEBUG_TTS | false | Debug TTS synthesis and playback |
| DEBUG_ECHO | false | Debug echo filter decisions |
| RATE | 16000 | Audio sample rate (Hz) |
| CHUNK | 1280 | Wake word detection chunk size (80ms) |
| VAD_CHUNK | 512 | Silero VAD chunk size (32ms) |
| VAD_THRESHOLD | 0.5 | Speech probability threshold |
| WAKE_WORD_NAME | jarvis | Wake word name (jarvis, marvin, mycroft) |
| WAKE_WORD_THRESHOLD | 0.9 | Wake word activation threshold |
| SILENCE_DURATION | 1.0s | Silence window to end recording |
| MAX_RECORD_SECONDS | 15 | Recording safety cap |
| CLAUDE_TIMEOUT | 300 | Claude API timeout (seconds) |
| STT_MODEL | medium.en | Faster-whisper model for STT |
| TTS_ENGINE | piper | Text-to-speech engine (piper or kokoro) |
| TTS_VOICE | en_US-lessac-medium | Voice model name (engine-specific) |
| TTS_OPENVINO_DEVICE | CPU | OpenVINO device for TTS (CPU or GPU) |
| INITIAL_ACK_DELAY | 10.0 | Seconds before first spoken filler |
| QUEUE_FILE | logs/prompt_queue.json | Persistent prompt queue for rate limit retries |

### Debug / Profiling Mode
Per-component debug flags in `[debug]` section of `jarvis.ini`. Each can be independently set to `true` to print timestamped `[DEBUG +<elapsed>s]` messages for that component:
- **models** — Per-model load times (wake word, whisper, VAD, TTS)
- **recording** — Recording duration, speech onset/offset, VAD decisions, full iteration timing
- **transcription** — WAV write, whisper inference, segment iteration breakdown
- **claude** — Subprocess startup, first event, tool_use events, total response time
- **tts** — TTS synthesis-to-first-audio latency and total playback time
- **echo** — Echo filter match decisions

Zero overhead when disabled (all debug paths are gated by per-component flags).

### ONNX Runtime Provider Auto-Detection
At startup, `_detect_onnx_provider()` checks for accelerated ONNX Runtime execution providers (currently OpenVINO for Intel CPUs). After config is loaded, `_patch_onnx_providers(device_type)` monkey-patches `onnxruntime.InferenceSession` to transparently inject the accelerated provider with the configured device type (`TTS_OPENVINO_DEVICE`) into all ONNX-based components. Models that are incompatible with the provider (e.g., openwakeword and silero-vad use unsupported `If` control flow ops) silently fall back to `CPUExecutionProvider`. Currently accelerated: Piper TTS, Kokoro TTS. C++ error output from failed probes is suppressed via stderr redirection.

The `TTS_OPENVINO_DEVICE` config option (default `CPU`) can be set to `GPU` to run TTS inference on an Intel integrated GPU via OpenVINO. GPU mode has a ~20s cold-start (graph compilation) but yields ~35% faster steady-state inference. The GPU session is created once via the monkey-patch and persists for the process lifetime. A warmup inference runs at startup to pre-compile the GPU graph. `_openvino_gpu_available()` checks for GPU device presence via the OpenVINO Core API.

## Dependencies
- **openwakeword**: Wake word detection (ONNX)
- **silero-vad**: Voice activity detection (ONNX)
- **faster-whisper**: Speech-to-text (CTranslate2)
- **piper-tts**: Text-to-speech (ONNX, default engine)
- **kokoro-onnx**: Text-to-speech (ONNX, alternative engine)
- **onnxruntime-openvino**: OpenVINO execution provider for ONNX Runtime (Intel CPU optimization)
- **numpy**: Audio array processing
- **torch**: Tensor ops for Silero VAD

## Preflight Validation (`tests/test_preflight.py`)

Safety gate between code changes and restart. Runs automatically in `jarvis.sh` after exit code 42, before spawning a new Jarvis process.

### Checks (ordered fastest-first)
1. **Syntax validation** — `py_compile` on `jarvis.py`. Also validates `config/jarvis.ini` exists and has all required sections.
2. **Config import** — Mirrors the exact import statement from `jarvis.py`.
3. **Config value validation** — Type checks, range checks (e.g., `0 < VAD_THRESHOLD <= 1.0`), non-empty message lists, TTS engine/voice validation.
4. **Module import** — `importlib` loads `jarvis.py` (catches broken imports, missing deps).
5. **Function signatures** — Verifies `load_models`, `record_until_silence`, `transcribe`, `is_self_echo`, `speak`, `send_to_claude`, `_always_on_listener`, `_strip_wake_prefix`, `main`, `parse_rate_limit`, queue functions exist with expected params.
5b. **Listener state** — Verifies `_wake_detected`, `_listener_suppress`, `_listener_needs_reset` module attributes exist.
5c. **ONNX provider detection** — Verifies `_detect_onnx_provider()` and `_openvino_gpu_available()` exist and return valid formats.
6. **Echo detection** — Verifies `is_self_echo()` correctly identifies echoes and passes through unrelated text.
7. **Rate limit parsing** — Verifies `parse_rate_limit()` extracts reset times from known formats and returns None for non-rate-limit text.
8. **Queue operations** — Tests `queue_add`, `queue_pop`, `queue_list`, `queue_clear` using a temp file.
9. **Model loading** — Loads all 4 models (wake word, whisper, VAD, TTS).

Exits 0 on pass, 1 on fail. All checks run (no early exit) to give a complete picture.

### Auto-revert flow (in `jarvis.sh`)
1. Jarvis exits with code 42 (restart requested).
2. `tests/test_preflight.py` is syntax-checked first, then run with a 120s timeout.
3. If preflight passes → restart normally.
4. If preflight fails → `git revert --no-edit HEAD`.
   - If revert succeeds, re-run preflight and restart.
   - If revert fails, fallback: `git checkout HEAD~1 -- jarvis.py config/jarvis.ini`.

## Emulation Testing (`tests/test_emulator.py`)

Replays synthetic interactions through the real Jarvis logic without hardware dependencies. Scenarios are defined as YAML files that script what each mock returns.

### How it works
- Imports `jarvis.py` and monkey-patches I/O classes at module boundaries (PulseRecorder, PulsePlayer, load_models, subprocess.Popen, subprocess.run, os._exit).
- `ScenarioDriver` loads a YAML file and coordinates mock behavior — each mock reads its scripted data from the current interaction.
- `jarvis.main()` runs with real logic (echo filtering, VAD silence detection, threading) against mocked I/O.
- When the scenario is exhausted, `MockPulseRecorder.read()` raises `JarvisExit` to cleanly exit.

### Mock classes (`tests/emulator/mocks.py`)
| Mock | Replaces | Behavior |
|---|---|---|
| MockPulseRecorder | PulseRecorder | Returns zero bytes; raises JarvisExit when exhausted |
| MockPulsePlayer | PulsePlayer | No-op writes; records audio events |
| MockWakeModel | openwakeword Model | Low scores for N chunks, then fires |
| MockVAD | Silero VAD | High prob for N speech_chunks, then low |
| MockWhisper | WhisperModel | Returns scripted transcription |
| MockTTS | PiperVoice | Records synthesized text, yields minimal audio |
| MockClaudeProcess | subprocess.Popen | Streams scripted JSON events with delay |

### Running
```
python tests/test_emulator.py                              # All scenarios
python tests/test_emulator.py --fast                       # 100x accelerated
python tests/test_emulator.py --debug                      # All debug flags
python tests/test_emulator.py tests/scenarios/X.yaml       # Single scenario
```

### Scenario YAML format
Each scenario defines a list of `interactions`, each with: `wake_word` (timing), `recording` (speech chunks), `transcription` (text), `claude` (delay, response, tool_events), and `expected` (validation checks). `time_scale` controls timing (1.0 = real-time, 0.01 = 100x faster).

### Dependencies
- **pyyaml**: YAML scenario file parsing

## Process Management
Runs as a systemd service. Exit code 42 triggers restart (used by built-in restart/revert commands and SIGUSR1). SIGINT triggers graceful shutdown with spoken goodbye.
