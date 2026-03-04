# Audio config
RATE = 16000
CHANNELS = 1
CHUNK = 1280  # 80ms at 16kHz - openwakeword expects this

# Silence detection config (Silero VAD)
VAD_THRESHOLD = 0.7            # speech probability threshold
VAD_CHUNK = 512                # Silero requires 512 samples at 16kHz
SILENCE_DURATION = 1.0         # seconds of silence after speech to stop
SILENCE_RATIO = 0.8            # fraction of silence window that must be quiet
MAX_RECORD_SECONDS = 15        # safety cap

# Wake word detection
WAKE_WORD_THRESHOLD = 0.8

# Claude API
CLAUDE_TIMEOUT = 300           # seconds before timing out Claude calls

# Spoken acknowledgements while waiting for Claude API response
ACKNOWLEDGEMENTS = [
    "Let me think about that.",
    "One moment.",
    "Working on it.",
    "Give me a second.",
    "On it.",
    "Let me look into that.",
    "Hmm, let me see.",
    "Just a moment.",
]

# Secondary acknowledgements for long-running requests (spoken every few seconds)
STILL_WORKING = [
    "Still working on it.",
    "Almost there.",
    "Still thinking.",
    "Hang on, still going.",
    "Bear with me.",
    "Still on it.",
]

STILL_WORKING_INTERVAL = 5  # seconds between secondary acknowledgements

# Startup and shutdown messages
STARTUP_MESSAGES = [
    "Jarvis is ready.",
    "At your service.",
    "Online and listening.",
    "Ready when you are.",
    "All systems go.",
    "Standing by.",
]

SHUTDOWN_MESSAGES = [
    "Shutting down. Goodbye.",
    "Going offline. See you later.",
    "Powering down. Take care.",
    "Signing off.",
    "Going to sleep. Goodbye.",
    "Until next time.",
]

# Initial acknowledgement delay (seconds before first filler is spoken)
INITIAL_ACK_DELAY = 10.0
