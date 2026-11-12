"""경로와 정해 둔 값."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # kidsrhythm-dev
AI_DIR = ROOT / "ai"
MODELS_DIR = Path(os.environ.get("KR_MODELS_DIR") or (AI_DIR / "models"))
SHERPA_DIR = MODELS_DIR / "sherpa"
SPEAKER_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
VAD_MODEL = "silero_vad.onnx"
SHERPA_URLS = {
    SPEAKER_MODEL: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + SPEAKER_MODEL,
    VAD_MODEL: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + VAD_MODEL,
}
TMP_DIR = Path(os.environ.get("KR_TMP_DIR") or (ROOT / "server" / "tmp"))
PID_FILE = ROOT / "agent" / "server.pid"
KEYS_FILE = ROOT / "ai-keys.env"

SR_IN = 16000  # 앱이 보내는 녹음 형식: PCM 16비트, 16kHz, 모노
CHUNK_MS = 2000
MAX_RECORDING_SEC = 180
MIN_RECORDING_SEC = 3.0
MIN_ENROLL_SPEECH_SEC = 3.0
KEEP_SEC = 600  # 세션·결과를 들고 있는 최대 시간 (10분)
JOB_TIMEOUT_SEC = 240
MAX_MEMBERS = 4
INSTRUMENTS = ("DRUM", "TAMBOURINE", "SHAKER", "BELL", "UKULELE", "STEELDRUM", "WOODBLOCK", "GUITAR", "HARP",
               "XYLOPHONE", "PIANO", "MARACAS")
STEPS = ("FIND_WORDS", "WRITE_LYRICS", "SPLIT_VOICES", "ADD_INSTRUMENTS", "FINISH")
