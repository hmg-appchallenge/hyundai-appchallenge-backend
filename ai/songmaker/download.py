"""모델을 미리 내려받는다 (음성 인식, TTS, 형태소 분석기)."""
from __future__ import annotations

import time
from pathlib import Path


def _size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0.0


def download(models: Path, stt_model: str, log) -> None:
    t0 = time.time()
    log(f"음성 인식 모델({stt_model}) 받는 중... 1.5GB쯤이라 몇 분 걸려.")
    from faster_whisper import WhisperModel

    WhisperModel(stt_model, device="cpu", compute_type="int8", download_root=str(models / "whisper"))
    log(f"  끝 ({time.time() - t0:.0f}초, {_size_gb(models / 'whisper'):.2f} GB)")

    t0 = time.time()
    log("TTS 모델(Supertonic-3) 받는 중...")
    from supertonic import TTS

    TTS(model="supertonic-3", model_dir=models / "supertonic3", auto_download=True)
    log(f"  끝 ({time.time() - t0:.0f}초, {_size_gb(models / 'supertonic3'):.2f} GB)")

    log("형태소 분석기(Kiwi) 확인 중...")
    from kiwipiepy import Kiwi

    Kiwi().tokenize("사과가 데굴데굴")
    log("  끝")
