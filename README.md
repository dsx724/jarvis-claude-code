# Jarvis

Voice assistant that listens for a wake word, records speech, transcribes it, sends it to Claude Code, and speaks the response aloud. Optionally accessible via Telegram bot.

## Prerequisites

- Linux with PulseAudio
- Python 3.10+
- A working microphone and speaker
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed and authenticated (`claude` must be on your PATH)

## Setup

```bash
./setup.sh
```

This installs system packages (`libpulse0`, `alsa-utils`, `ffmpeg`), creates a Python virtualenv, and installs all Python dependencies. It's idempotent and safe to re-run.

## Configuration

All settings live in `config/jarvis.ini`. Defaults work out of the box for the voice assistant — no edits required to get started.

### Telegram bot (optional)

To enable the Telegram bot, create `config/secrets.ini` (gitignored):

```ini
[telegram]
bot_token = YOUR_BOT_TOKEN
allowed_users = 123456789,987654321
```

- `bot_token` — get one from [@BotFather](https://t.me/BotFather) on Telegram
- `allowed_users` — comma-separated Telegram user IDs that are allowed to interact with the bot (leave empty to allow all)

## Running

Manually:

```bash
./jarvis.sh
```

As a systemd user service:

```bash
cp jarvis.service ~/.config/systemd/user/
systemctl --user enable --now jarvis
```

## Built-in voice commands

These are handled locally without calling Claude:

| Command | Action |
|---|---|
| shutdown / shut down / go to sleep / goodnight | Stops Jarvis (clean exit) |
| restart / reboot | Restarts Jarvis |
| revert | Reverts the last git commit and restarts |
| stop / cancel / never mind | Kills Claude mid-processing |
| queue / show queue | Lists queued prompts |
| clear queue / empty queue | Clears the prompt queue |

## Licensing

Jarvis source code is licensed under **GPL-3.0** (see [LICENSE](LICENSE)). GPL-3.0 allows commercial use — it only requires sharing source code when you distribute the software to others.

Some **pre-trained models** bundled with or downloaded by Jarvis have separate, more restrictive licenses:

| Component | License | Commercial use? |
|---|---|---|
| Jarvis source code | GPL-3.0 | Yes |
| piper-tts (library) | GPL-3.0 | Yes |
| Kokoro TTS model & voices | Apache-2.0 | Yes |
| OpenAI Whisper model | MIT | Yes |
| Silero VAD model | MIT | Yes |
| openwakeword (code) | Apache-2.0 | Yes |
| openwakeword pre-trained models | CC BY-NC-SA 4.0 | **No** |
| Piper voices (varies by voice) | See `jarvis.ini` | Some yes, some no |

### Commercial use

To use Jarvis commercially, set `commercial_use = true` in `config/jarvis.ini` under `[licensing]`. Jarvis will validate at startup that all components have commercially-compatible licenses and exit with clear errors if not.

You will need to:

1. **TTS** — use the `kokoro` engine (Apache-2.0), or a Piper voice marked `[commercial]` in `jarvis.ini` (e.g. `en_US-ljspeech-medium`)
2. **Wake word** — provide a custom-trained model via `[wake_word] model_path` (the openwakeword training code is Apache-2.0, so models you train are yours)

### Contributing

Contributions are welcome. By submitting a pull request, you agree to the [Contributor License Agreement](CLA.md), which grants the maintainer the right to use your contribution under both the GPL-3.0 public license and a separate commercial license.

## Project structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed internals.

## Testing

```bash
# Run all emulation scenarios
python tests/test_emulator.py

# Fast mode (100x accelerated)
python tests/test_emulator.py --fast

# Single scenario
python tests/test_emulator.py tests/scenarios/builtin_restart.yaml
```
