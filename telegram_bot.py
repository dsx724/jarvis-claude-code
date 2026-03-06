#!/usr/bin/env python3
"""Jarvis Telegram Bot — text and voice interface to Claude Code."""

import asyncio
import configparser
import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_cfg = configparser.ConfigParser()
_cfg.read(os.path.join(SCRIPT_DIR, "config", "jarvis.ini"))

TELEGRAM_TOKEN = _cfg.get("telegram", "bot_token", fallback="")
ALLOWED_USERS = _cfg.get("telegram", "allowed_users", fallback="")
ALLOWED_USER_IDS = {int(uid.strip()) for uid in ALLOWED_USERS.split(",") if uid.strip()} if ALLOWED_USERS else set()
VOICE_REPLIES = _cfg.getboolean("telegram", "voice_replies", fallback=True)

STT_MODEL = _cfg.get("stt", "model")
TTS_ENGINE = _cfg.get("tts", "engine")
TTS_VOICE = _cfg.get("tts", "voice")
CLAUDE_TIMEOUT = _cfg.getint("claude", "timeout")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("jarvis.telegram")

conversation_logger = logging.getLogger("jarvis.conversation")
conversation_logger.setLevel(logging.INFO)
_conv_handler = logging.FileHandler(os.path.join(SCRIPT_DIR, "logs", "conversation.log"))
_conv_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
conversation_logger.addHandler(_conv_handler)

# ---------------------------------------------------------------------------
# Claude integration (reuses same approach as jarvis.py)
# ---------------------------------------------------------------------------

SESSION_ID = str(uuid.uuid4())


def send_to_claude(text, first_call=[True]):
    """Send text to Claude Code and return the response."""
    log.info("Sending to Claude: %s", text[:80])
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "-p", "--dangerously-skip-permissions",
               "--verbose", "--output-format", "stream-json"]
        if first_call[0]:
            cmd += ["--session-id", SESSION_ID]
            first_call[0] = False
        else:
            cmd += ["--resume", SESSION_ID]
        cmd.append(text)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )

        timer = threading.Timer(CLAUDE_TIMEOUT, proc.kill)
        timer.start()

        response = None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    response = event.get("result", "").strip()
            proc.wait()
        finally:
            timer.cancel()

        if response is not None:
            if proc.returncode != 0 and not response:
                stderr = proc.stderr.read().strip()
                return f"Error: {stderr}"
            return response

        stderr = proc.stderr.read().strip()
        if proc.returncode != 0:
            return f"Error: {stderr}" if stderr else "Error: Claude returned no response."
        return "Error: Claude returned no response."

    except FileNotFoundError:
        return "Error: 'claude' command not found. Is Claude Code installed?"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

whisper_model = None
tts_voice = None


def load_models():
    global whisper_model, tts_voice

    log.info("Loading Whisper model (%s)...", STT_MODEL)
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")

    if VOICE_REPLIES:
        log.info("Loading TTS model (%s: %s)...", TTS_ENGINE, TTS_VOICE)
        if TTS_ENGINE == "piper":
            from piper import PiperVoice
            voice_dir = os.path.join(SCRIPT_DIR, "voices")
            voice_path = os.path.join(voice_dir, f"{TTS_VOICE}.onnx")
            voice_config = voice_path + ".json"
            if not os.path.exists(voice_path):
                os.makedirs(voice_dir, exist_ok=True)
                log.info("Downloading TTS voice (%s)...", TTS_VOICE)
                import urllib.request
                parts = TTS_VOICE.split("-")
                lang = parts[0]
                lang_family = lang.split("_")[0]
                dataset = parts[1]
                quality = parts[2]
                base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang}/{dataset}/{quality}"
                urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx", voice_path)
                urllib.request.urlretrieve(f"{base}/{TTS_VOICE}.onnx.json", voice_config)
            tts_voice = PiperVoice.load(voice_path)
        elif TTS_ENGINE == "kokoro":
            from kokoro_onnx import Kokoro
            voice_dir = os.path.join(SCRIPT_DIR, "voices")
            model_path = os.path.join(voice_dir, "kokoro-v1.0.onnx")
            voices_path = os.path.join(voice_dir, "voices-v1.0.bin")
            if not os.path.exists(model_path) or not os.path.exists(voices_path):
                raise FileNotFoundError(
                    f"Kokoro model files not found in {voice_dir}. "
                    "Download kokoro-v1.0.onnx and voices-v1.0.bin from "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases"
                )
            kokoro = Kokoro(model_path, voices_path)

            class _Config:
                sample_rate = 24000

            class KokoroTTS:
                config = _Config()
                def synthesize(self, text):
                    samples, _rate = kokoro.create(text, voice=TTS_VOICE, speed=1.0)
                    yield type('Chunk', (), {'audio_float_array': samples.astype(np.float32)})()

            tts_voice = KokoroTTS()
        else:
            raise ValueError(f"Unsupported TTS engine: {TTS_ENGINE}")

    log.info("Models loaded.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def ogg_to_wav(ogg_bytes):
    """Convert OGG/Opus audio to 16kHz mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_f:
        ogg_f.write(ogg_bytes)
        ogg_path = ogg_f.name
    wav_path = ogg_path.replace(".ogg", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True,
        )
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        for p in (ogg_path, wav_path):
            if os.path.exists(p):
                os.unlink(p)


def transcribe(audio_wav_bytes):
    """Transcribe WAV audio bytes with faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_wav_bytes)
        tmp_path = f.name
    try:
        segments, _info = whisper_model.transcribe(
            tmp_path, beam_size=1, without_timestamps=True, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


def synthesize_ogg(text):
    """Synthesize text to OGG/Opus bytes for Telegram voice messages."""
    if tts_voice is None:
        return None

    # Collect raw PCM from TTS
    pcm_chunks = []
    for chunk in tts_voice.synthesize(text):
        audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
        pcm_chunks.append(audio_int16.tobytes())
    pcm_data = b"".join(pcm_chunks)

    # Write to temp WAV, then convert to OGG with ffmpeg
    sample_rate = tts_voice.config.sample_rate
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        with wave.open(wav_f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        wav_path = wav_f.name

    ogg_path = wav_path.replace(".wav", ".ogg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", ogg_path],
            capture_output=True, check=True,
        )
        with open(ogg_path, "rb") as f:
            return f.read()
    finally:
        for p in (wav_path, ogg_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Telegram bot handlers
# ---------------------------------------------------------------------------

def check_allowed(user_id):
    """Return True if the user is allowed (or if no allowlist is configured)."""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


async def handle_start(update, context):
    if not check_allowed(update.effective_user.id):
        return
    await update.message.reply_text("Jarvis online. Send me a text or voice message.")


async def handle_text(update, context):
    user = update.effective_user
    if not check_allowed(user.id):
        return

    text = update.message.text.strip()
    if not text:
        return

    log.info("Text from %s (%d): %s", user.first_name, user.id, text[:80])
    conversation_logger.info("USER (telegram/%d): %s", user.id, text)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, send_to_claude, text)

    conversation_logger.info("CLAUDE (telegram/%d): %s", user.id, response)
    log.info("Response: %s", response[:80])

    await update.message.reply_text(response)

    if VOICE_REPLIES and tts_voice is not None:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
        ogg_data = await loop.run_in_executor(None, synthesize_ogg, response)
        if ogg_data:
            await update.message.reply_voice(voice=io.BytesIO(ogg_data))


async def handle_voice(update, context):
    user = update.effective_user
    if not check_allowed(user.id):
        return

    log.info("Voice from %s (%d)", user.first_name, user.id)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    voice_file = await update.message.voice.get_file()
    ogg_bytes = await voice_file.download_as_bytearray()

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, ogg_to_wav, bytes(ogg_bytes))
    text = await loop.run_in_executor(None, transcribe, wav_bytes)

    if not text:
        await update.message.reply_text("(couldn't transcribe audio)")
        return

    log.info("Transcribed: %s", text[:80])
    conversation_logger.info("USER (telegram-voice/%d): %s", user.id, text)

    # Show the user what was transcribed
    await update.message.reply_text(f"_{text}_", parse_mode="Markdown")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = await loop.run_in_executor(None, send_to_claude, text)

    conversation_logger.info("CLAUDE (telegram/%d): %s", user.id, response)
    log.info("Response: %s", response[:80])

    await update.message.reply_text(response)

    if VOICE_REPLIES and tts_voice is not None:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
        ogg_data = await loop.run_in_executor(None, synthesize_ogg, response)
        if ogg_data:
            await update.message.reply_voice(voice=io.BytesIO(ogg_data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_TOKEN:
        print("Error: Set bot_token in [telegram] section of config/jarvis.ini")
        print("Get a token from @BotFather on Telegram.")
        return

    os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)
    load_models()

    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    log.info("Telegram bot starting (session %s)...", SESSION_ID)
    if ALLOWED_USER_IDS:
        log.info("Allowed users: %s", ALLOWED_USER_IDS)
    else:
        log.info("No user allowlist — accepting messages from everyone")
    app.run_polling()


if __name__ == "__main__":
    main()
