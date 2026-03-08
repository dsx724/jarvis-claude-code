#!/usr/bin/env python3
"""Jarvis Telegram Bot — text and voice interface to Claude Code.

Can run standalone (python telegram_bot.py) or be started as a background
thread from jarvis.py by calling start_telegram_bot().
"""

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import threading
import wave

import numpy as np

log = logging.getLogger("jarvis.telegram")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def synthesize_ogg(tts_voice, text, tts_lock=None):
    """Synthesize text to OGG/Opus bytes for Telegram voice messages."""
    if tts_voice is None:
        return None

    # Serialize TTS access — the ONNX session (especially OpenVINO GPU) is not
    # thread-safe, so we share a lock with the voice assistant's speak().
    lock = tts_lock or threading.Lock()
    with lock:
        pcm_chunks = []
        for chunk in tts_voice.synthesize(text):
            audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
            pcm_chunks.append(audio_int16.tobytes())
        pcm_data = b"".join(pcm_chunks)
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
# Bot setup and handlers
# ---------------------------------------------------------------------------

def _build_app(token, allowed_user_ids, voice_replies, whisper_model,
               tts_voice, send_to_claude_fn, conversation_logger, tts_lock=None):
    """Build and return a configured Telegram Application."""
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    def check_allowed(user_id):
        if not allowed_user_ids:
            return True
        return user_id in allowed_user_ids

    def _check_reconnected():
        if _disconnected[0]:
            log.info("Telegram reconnected.")
            _disconnected[0] = False

    async def handle_start(update, context):
        _check_reconnected()
        if not check_allowed(update.effective_user.id):
            return
        await update.message.reply_text("Jarvis online. Send me a text or voice message.")

    async def handle_text(update, context):
        _check_reconnected()
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
        response = await loop.run_in_executor(None, send_to_claude_fn, text)

        conversation_logger.info("CLAUDE (telegram/%d): %s", user.id, response)
        log.info("Response: %s", response[:80])

        await update.message.reply_text(response)

        if voice_replies and tts_voice is not None:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
            ogg_data = await loop.run_in_executor(None, synthesize_ogg, tts_voice, response, tts_lock)
            if ogg_data:
                await update.message.reply_voice(voice=io.BytesIO(ogg_data))

    async def handle_voice(update, context):
        _check_reconnected()
        user = update.effective_user
        if not check_allowed(user.id):
            return

        log.info("Voice from %s (%d)", user.first_name, user.id)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        voice_file = await update.message.voice.get_file()
        ogg_bytes = await voice_file.download_as_bytearray()

        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, ogg_to_wav, bytes(ogg_bytes))

        # Transcribe using the shared Whisper model
        def _transcribe(wav_data):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                tmp_path = f.name
            try:
                segments, _info = whisper_model.transcribe(
                    tmp_path, beam_size=1, without_timestamps=True, vad_filter=True
                )
                return " ".join(seg.text.strip() for seg in segments).strip()
            finally:
                os.unlink(tmp_path)

        text = await loop.run_in_executor(None, _transcribe, wav_bytes)

        if not text:
            await update.message.reply_text("(couldn't transcribe audio)")
            return

        log.info("Transcribed: %s", text[:80])
        conversation_logger.info("USER (telegram-voice/%d): %s", user.id, text)

        await update.message.reply_text(f"_{text}_", parse_mode="Markdown")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        response = await loop.run_in_executor(None, send_to_claude_fn, text)

        conversation_logger.info("CLAUDE (telegram/%d): %s", user.id, response)
        log.info("Response: %s", response[:80])

        await update.message.reply_text(response)

        if voice_replies and tts_voice is not None:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
            ogg_data = await loop.run_in_executor(None, synthesize_ogg, tts_voice, response, tts_lock)
            if ogg_data:
                await update.message.reply_voice(voice=io.BytesIO(ogg_data))

    _disconnected = [False]

    async def error_handler(update, context):
        """Handle errors from Telegram bot handlers."""
        from telegram.error import NetworkError, TimedOut
        err = context.error
        if isinstance(err, (NetworkError, TimedOut)):
            if not _disconnected[0]:
                log.warning("Telegram connection error: %s", err)
                _disconnected[0] = True
            return
        log.exception("Telegram bot error: %s", err)
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "Sorry, something went wrong processing your request."
                )
            except Exception:
                pass

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_error_handler(error_handler)
    return app


def start_telegram_bot(token, allowed_user_ids, voice_replies, whisper_model,
                       tts_voice, send_to_claude_fn, conversation_logger,
                       tts_lock=None):
    """Start the Telegram bot as a background daemon thread.

    Called from jarvis.py main() to run alongside the voice interface.
    """
    if not token:
        log.info("No Telegram bot_token configured — skipping Telegram bot.")
        return

    try:
        from telegram.ext import ApplicationBuilder
    except ImportError:
        log.warning("python-telegram-bot not installed — skipping Telegram bot. "
                     "Install with: pip install python-telegram-bot")
        return

    def _run():
        # Suppress the library's own verbose traceback logging for network errors.
        # Our error_handler handles these with clean one-line messages instead.
        logging.getLogger("telegram.ext._utils.networkloop").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        app = _build_app(token, allowed_user_ids, voice_replies, whisper_model,
                         tts_voice, send_to_claude_fn, conversation_logger,
                         tts_lock=tts_lock)
        log.info("Telegram bot starting...")
        if allowed_user_ids:
            log.info("Allowed Telegram users: %s", allowed_user_ids)
        else:
            log.info("No Telegram user allowlist — accepting messages from everyone")

        # Can't use run_polling() from a non-main thread because it tries to
        # register signal handlers. Manage the asyncio loop manually instead.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(app.initialize())
            loop.run_until_complete(app.updater.start_polling())
            loop.run_until_complete(app.start())
            loop.run_forever()
        except Exception:
            log.exception("Telegram bot error")
        finally:
            loop.run_until_complete(app.updater.stop())
            loop.run_until_complete(app.stop())
            loop.run_until_complete(app.shutdown())
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import configparser
    import uuid

    logging.basicConfig(format="%(asctime)s [%(name)s] %(message)s", level=logging.INFO)

    _cfg = configparser.ConfigParser()
    _cfg.read([
        os.path.join(SCRIPT_DIR, "config", "jarvis.ini"),
        os.path.join(SCRIPT_DIR, "config", "secrets.ini"),
    ])

    token = _cfg.get("telegram", "bot_token", fallback="")
    if not token:
        print("Error: Set bot_token in [telegram] section of config/secrets.ini")
        print("Get a token from @BotFather on Telegram.")
        raise SystemExit(1)

    allowed_raw = _cfg.get("telegram", "allowed_users", fallback="")
    allowed_ids = {int(uid.strip()) for uid in allowed_raw.split(",") if uid.strip()} if allowed_raw else set()
    voice_replies = _cfg.getboolean("telegram", "voice_replies", fallback=True)

    stt_model = _cfg.get("stt", "model")
    tts_engine = _cfg.get("tts", "engine")
    tts_voice_name = _cfg.get("tts", "voice")
    claude_timeout = _cfg.getint("claude", "timeout")

    os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)

    conv_logger = logging.getLogger("jarvis.conversation")
    conv_logger.setLevel(logging.INFO)
    _h = logging.FileHandler(os.path.join(SCRIPT_DIR, "logs", "conversation.log"))
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    conv_logger.addHandler(_h)

    # Load models
    log.info("Loading Whisper model (%s)...", stt_model)
    from faster_whisper import WhisperModel
    w_model = WhisperModel(stt_model, device="cpu", compute_type="int8")

    t_voice = None
    if voice_replies:
        log.info("Loading TTS model (%s: %s)...", tts_engine, tts_voice_name)
        if tts_engine == "piper":
            from piper import PiperVoice
            voice_dir = os.path.join(SCRIPT_DIR, "voices")
            voice_path = os.path.join(voice_dir, f"{tts_voice_name}.onnx")
            t_voice = PiperVoice.load(voice_path)
        elif tts_engine == "kokoro":
            from kokoro_onnx import Kokoro
            voice_dir = os.path.join(SCRIPT_DIR, "voices")
            kokoro = Kokoro(
                os.path.join(voice_dir, "kokoro-v1.0.onnx"),
                os.path.join(voice_dir, "voices-v1.0.bin"),
            )
            class _Cfg:
                sample_rate = 24000
            class _KokoroTTS:
                config = _Cfg()
                def synthesize(self, text):
                    samples, _ = kokoro.create(text, voice=tts_voice_name, speed=1.0)
                    yield type('C', (), {'audio_float_array': samples.astype(np.float32)})()
            t_voice = _KokoroTTS()

    # Standalone Claude sender
    _session_id = str(uuid.uuid4())
    import json

    def _send_to_claude(text, _first=[True]):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "-p", "--dangerously-skip-permissions",
               "--verbose", "--output-format", "stream-json"]
        if _first[0]:
            cmd += ["--session-id", _session_id]
            _first[0] = False
        else:
            cmd += ["--resume", _session_id]
        cmd.append(text)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        timer = threading.Timer(claude_timeout, proc.kill)
        timer.start()
        response = None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if event.get("type") == "result":
                    response = event.get("result", "").strip()
            proc.wait()
        finally:
            timer.cancel()
        if response:
            return response
        stderr = proc.stderr.read().strip()
        return f"Error: {stderr}" if stderr else "Error: Claude returned no response."

    app = _build_app(token, allowed_ids, voice_replies, w_model, t_voice,
                     _send_to_claude, conv_logger)
    log.info("Telegram bot starting (standalone, session %s)...", _session_id)
    app.run_polling()
