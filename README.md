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
