#!/usr/bin/env python3
"""Jarvis: Wake-word voice assistant for Claude Code."""

import os
import signal
import subprocess
import sys
import tempfile
import uuid
import wave

import numpy as np
import pyaudio

# Audio config
RATE = 16000
CHANNELS = 1
CHUNK = 1280  # 80ms at 16kHz — openwakeword expects this
FORMAT = pyaudio.paInt16

# Silence detection config
NOISE_CALIBRATION_SECONDS = 2  # how long to sample ambient noise
NOISE_MULTIPLIER = 3.0         # speech threshold = ambient_rms * this
SILENCE_DURATION = 1.5         # seconds of silence after speech to stop
MAX_RECORD_SECONDS = 30        # safety cap


def load_models():
    """Load wake word and whisper models."""
    print("Loading wake word model...")
    from openwakeword.model import Model as WakeModel
    wake_model = WakeModel(wakeword_model_paths=[
        os.path.join(os.path.dirname(__import__('openwakeword').__file__),
                     "resources", "models", "hey_jarvis_v0.1.onnx")
    ])

    print("Loading whisper model (base.en)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    print("Loading TTS model...")
    from piper import PiperVoice
    # Download voice if needed
    voice_dir = os.path.join(os.path.dirname(__file__), "voices")
    voice_path = os.path.join(voice_dir, "en_US-lessac-medium.onnx")
    voice_config = voice_path + ".json"

    if not os.path.exists(voice_path):
        os.makedirs(voice_dir, exist_ok=True)
        print("Downloading TTS voice...")
        import urllib.request
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
        urllib.request.urlretrieve(f"{base}/en_US-lessac-medium.onnx", voice_path)
        urllib.request.urlretrieve(f"{base}/en_US-lessac-medium.onnx.json", voice_config)

    tts_voice = PiperVoice.load(voice_path)

    print("All models loaded.")
    return wake_model, whisper_model, tts_voice


def calibrate_noise(stream):
    """Measure ambient noise RMS over a few seconds."""
    print("Calibrating ambient noise...")
    chunks_per_second = RATE / CHUNK
    num_chunks = int(NOISE_CALIBRATION_SECONDS * chunks_per_second)
    rms_values = []

    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms_values.append(np.sqrt(np.mean(audio_data ** 2)))

    ambient_rms = np.mean(rms_values)
    threshold = max(ambient_rms * NOISE_MULTIPLIER, 200)  # floor of 200
    print(f"Ambient RMS: {ambient_rms:.0f}, speech threshold: {threshold:.0f}")
    return threshold


def record_until_silence(stream, speech_threshold):
    """Record audio until speech is followed by silence. Returns raw audio bytes."""
    print("Listening... (speak now)")
    frames = []
    silent_chunks = 0
    speech_detected = False
    chunks_per_second = RATE / CHUNK
    silence_chunks_needed = int(SILENCE_DURATION * chunks_per_second)

    for _ in range(int(MAX_RECORD_SECONDS * chunks_per_second)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms >= speech_threshold:
            speech_detected = True
            silent_chunks = 0
        else:
            if speech_detected:
                silent_chunks += 1

        # Only stop after speech was detected and then silence followed
        if speech_detected and silent_chunks >= silence_chunks_needed:
            break

    print("Done recording.")
    return b"".join(frames)


def transcribe(whisper_model, audio_bytes):
    """Transcribe raw audio bytes with faster-whisper."""
    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)

    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    finally:
        os.unlink(tmp_path)


def speak(tts_voice, text):
    """Speak text aloud using piper TTS."""
    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(tts_voice.config.sample_rate)
            for chunk in tts_voice.synthesize(text):
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

    try:
        subprocess.run(["aplay", "-q", tmp_path], check=False)
    finally:
        os.unlink(tmp_path)


SESSION_ID = str(uuid.uuid4())


def send_to_claude(text, first_call=[True]):
    """Send text to Claude Code and return the response, resuming the session."""
    print(f"\n> {text}\n")
    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "-p", "--dangerously-skip-permissions"]
        if first_call[0]:
            cmd += ["--session-id", SESSION_ID]
            first_call[0] = False
        else:
            cmd += ["--resume", SESSION_ID]
        cmd.append(text)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        response = result.stdout.strip()
        if result.returncode != 0 and not response:
            response = f"Error: {result.stderr.strip()}"
        return response
    except FileNotFoundError:
        return "Error: 'claude' command not found. Is Claude Code installed?"
    except subprocess.TimeoutExpired:
        return "Error: Claude Code timed out."


def main():
    wake_model, whisper_model, tts_voice = load_models()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    speech_threshold = calibrate_noise(stream)

    def shutdown(sig, frame):
        print("\nShutting down...")
        os._exit(0)

    def restart(sig, frame):
        print("\nRestarting...")
        os._exit(42)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGUSR1, restart)

    print("\n=== Jarvis is ready. Say 'Hey Jarvis' to activate. ===\n")

    while True:
        # Read audio chunk for wake word detection
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Feed to wake word detector
        prediction = wake_model.predict(audio_data)

        # Check for wake word activation
        for model_name, score in prediction.items():
            if score > 0.5:
                print("\n*** Wake word detected! ***")

                # Record until silence
                audio_bytes = record_until_silence(stream, speech_threshold)

                # Transcribe
                text = transcribe(whisper_model, audio_bytes)
                if not text:
                    print("(no speech detected)")
                    continue

                print(f"Transcribed: {text}")

                # Send to Claude
                response = send_to_claude(text)
                print(f"\nClaude: {response}\n")

                # Speak response
                speak(tts_voice, response)

                # Reset wake word model state
                wake_model.reset()


if __name__ == "__main__":
    main()
