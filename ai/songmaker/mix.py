"""반주와 보컬을 섞어 스테레오 mp3로 만든다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from .dsp import compress, db_to_amp, envelope, filt, limit, pan, reverb_ir

# 줄기별 음량(dB)과 좌우 위치. 가족 타악기는 좌석처럼 왼쪽, 오른쪽, 가운데에 둔다.
LEVELS = {
    "kick": (-10.0, 0.0), "snare": (-3.0, 0.05), "hat": (-8.0, 0.3), "clap": (-3.0, -0.1),
    "crash": (-12.0, 0.2), "bass": (-16.0, 0.0), "inst": (-4.0, -0.3), "melody": (-4.0, 0.2),
    "mem0": (-16.0, -0.55), "mem1": (-7.0, 0.55), "mem2": (-12.0, 0.0),
}
DUCKED = {"inst", "melody", "bass"}
VOCAL_ACTIVE_DB = -16.0  # 소리 나는 구간 기준 보컬 음량


def active_normalize(x: np.ndarray, sr: int, target_db: float) -> np.ndarray:
    """소리가 나는 구간만 보고 음량을 맞춘다 (드문드문한 보컬도 같은 크기로)."""
    if not np.any(x):
        return x
    env = envelope(x, sr, 20.0)
    active = env > env.max() * db_to_amp(-35.0)
    level = np.sqrt(np.mean(np.square(x[active], dtype=np.float64)) + 1e-12)
    y = x * (db_to_amp(target_db) / level)
    pk = float(np.max(np.abs(y)))
    return (y * (0.98 / pk) if pk > 0.98 else y).astype(np.float32)


def mix(stems: dict[str, np.ndarray], vocal: np.ndarray, sr: int, variant: str) -> np.ndarray:
    n = len(vocal)
    out = np.zeros((n, 2), np.float32)
    # 보컬: 압축 -> 리버브
    v = compress(vocal, sr, -22.0, 3.0, 90.0)
    v = active_normalize(v, sr, VOCAL_ACTIVE_DB)
    venv = envelope(v, sr, 60.0)
    venv = venv / (venv.max() + 1e-9)
    duck = 1.0 - 0.45 * np.clip(venv * 1.6, 0.0, 1.0)  # 최대 약 5 dB 줄임
    for name, x in stems.items():
        if name not in LEVELS or not np.any(x):
            continue
        db, p = LEVELS[name]
        if variant == "A" and name == "melody":
            db += 4.0  # 아이 목소리만 있는 버전은 멜로디를 조금 더 살린다
        y = x[:n] * db_to_amp(db)
        if name in DUCKED:
            y = y * duck[: len(y)]
        out[: len(y)] += pan(y, p)
    out += pan(v, 0.0)
    irl, irr = reverb_ir(sr, 1.1, 0.9, 11), reverb_ir(sr, 1.1, 0.9, 12)
    send = filt(v, sr, "highpass", 200) * db_to_amp(-15.0)
    wet = np.stack([signal.fftconvolve(send, irl)[:n], signal.fftconvolve(send, irr)[:n]], axis=1)
    inst_send = filt(stems.get("inst", np.zeros(n))[:n] * db_to_amp(-24.0), sr, "highpass", 300)
    wet += np.stack([signal.fftconvolve(inst_send, irl)[:n], signal.fftconvolve(inst_send, irr)[:n]], axis=1)
    out += wet.astype(np.float32)
    # 마스터: 저역 정리 -> 음량 맞춤 -> 리미터
    out = np.stack([filt(out[:, 0], sr, "highpass", 30), filt(out[:, 1], sr, "highpass", 30)], axis=1)
    level = np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-12)
    out = out * (db_to_amp(-14.5) / level)
    return limit(out, sr, -1.5, 3.0, 80.0)


def save_mp3(st: np.ndarray, sr: int, path: Path, kbps: int = 192) -> None:
    import lameenc

    pcm = (np.clip(st, -1.0, 1.0) * 32767.0).astype("<i2")
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(2)
    data = enc.encode(pcm.tobytes()) + enc.flush()
    Path(path).write_bytes(data)


def save_wav(st: np.ndarray, sr: int, path: Path) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(st, -1.0, 1.0), sr, subtype="PCM_16")
