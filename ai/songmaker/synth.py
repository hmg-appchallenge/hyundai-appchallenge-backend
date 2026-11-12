"""악기 소리를 샘플 파일 없이 직접 합성한다."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

from .dsp import filt

# 계절별 반주 악기 (메모리 기획: 봄 우쿨렐레, 여름 스틸드럼, 가을 어쿠스틱, 겨울 벨)
SEASON_SETS = {
    "spring": "ukulele",
    "summer": "steel_drum",
    "autumn": "acoustic_guitar",
    "winter": "bell",
}
SET_LABELS = {
    "ukulele": "우쿨렐레",
    "steel_drum": "스틸드럼",
    "acoustic_guitar": "어쿠스틱 기타",
    "bell": "벨",
}
# 가족 한 명당 타악기 하나
MEMBER_INSTRUMENTS = ["drum", "tambourine", "shaker"]
MEMBER_LABELS = {"drum": "북", "tambourine": "탬버린", "shaker": "셰이커"}


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _t(sec: float, sr: int) -> np.ndarray:
    return np.arange(int(sec * sr)) / sr


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _norm(y: np.ndarray, p: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return (y * (p / m)).astype(np.float32) if m > 0 else y.astype(np.float32)


def _tail_fade(y: np.ndarray, sr: int, sec: float = 0.03) -> np.ndarray:
    n = min(len(y), int(sec * sr))
    if n > 1:
        y = y.copy()
        y[-n:] *= np.linspace(1.0, 0.0, n)
    return y


# ---------- 드럼 ----------
@lru_cache(maxsize=None)
def kick(sr: int) -> np.ndarray:
    t = _t(0.5, sr)
    f = 46 + 120 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 7)
    click = filt(_noise(len(t), 1), sr, "highpass", 1500) * np.exp(-t * 350) * 0.25
    return _norm(body + click)


@lru_cache(maxsize=None)
def snare(sr: int) -> np.ndarray:
    t = _t(0.3, sr)
    noise = filt(_noise(len(t), 2), sr, "bandpass", [1200, 9000]) * np.exp(-t * 18)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 20) * 0.6 + np.sin(2 * np.pi * 330 * t) * np.exp(-t * 25) * 0.3
    return _norm(noise * 0.9 + tone)


@lru_cache(maxsize=None)
def clap(sr: int) -> np.ndarray:
    t = _t(0.35, sr)
    base = filt(_noise(len(t), 3), sr, "bandpass", [900, 4000])
    env = np.zeros_like(t)
    for k, d in enumerate((0.0, 0.011, 0.022)):
        env += (t >= d) * np.exp(-np.maximum(t - d, 0) * 180) * (0.8 + 0.1 * k)
    env += (t >= 0.03) * np.exp(-np.maximum(t - 0.03, 0) * 14) * 0.5
    return _norm(base * env)


@lru_cache(maxsize=None)
def hat(sr: int, open_: bool = False) -> np.ndarray:
    t = _t(0.4 if open_ else 0.08, sr)
    metal = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (3140, 4730, 5650, 6920, 8200, 9370))
    noise = _noise(len(t), 4)
    y = filt(metal * 0.3 + noise, sr, "highpass", 7000)
    return _norm(y * np.exp(-t * (7 if open_ else 55)))


@lru_cache(maxsize=None)
def crash(sr: int) -> np.ndarray:
    t = _t(2.0, sr)
    y = filt(_noise(len(t), 5), sr, "highpass", 3500) * np.exp(-t * 2.0) * (1 - np.exp(-t * 400))
    return _norm(y)


# ---------- 가족 타악기 ----------
@lru_cache(maxsize=None)
def tom_tuned(sr: int, f0: float = 105.0) -> np.ndarray:
    t = _t(0.45, sr)
    f = f0 * (1 + 0.45 * np.exp(-t * 18))
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 7.5)
    skin = filt(_noise(len(t), 6), sr, "bandpass", [200, 1500]) * np.exp(-t * 40) * 0.25
    return _norm(body + skin)


def tom(sr: int) -> np.ndarray:
    return tom_tuned(sr, 105.0)


@lru_cache(maxsize=None)
def tambourine(sr: int) -> np.ndarray:
    t = _t(0.3, sr)
    jingle = sum(np.sin(2 * np.pi * f * t) for f in (6800, 8150, 9600, 11200)) * np.exp(-t * 22)
    noise = filt(_noise(len(t), 7), sr, "bandpass", [5000, 12000]) * np.exp(-t * 28)
    env = 1 - np.exp(-t * 900)
    return _norm((jingle * 0.4 + noise) * env)


@lru_cache(maxsize=None)
def shaker(sr: int) -> np.ndarray:
    t = _t(0.13, sr)
    noise = filt(_noise(len(t), 8), sr, "bandpass", [3500, 9500])
    env = np.where(t < 0.03, t / 0.03, np.exp(-(t - 0.03) * 35))
    return _norm(noise * env)


@lru_cache(maxsize=None)
def woodblock(sr: int) -> np.ndarray:
    """우드블록: 속이 빈 나무 토막을 친 짧고 단단한 '똑' 소리."""
    t = _t(0.18, sr)
    body = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((1180, 1.0, 38), (2650, 0.35, 60), (4300, 0.12, 90)))
    knock = filt(_noise(len(t), 11), sr, "bandpass", [1500, 6000]) * np.exp(-t * 400) * 0.3
    return _norm((body + knock) * (1 - np.exp(-t * 3000)))


@lru_cache(maxsize=None)
def maracas(sr: int) -> np.ndarray:
    """마라카스: 씨앗이 부딪치는 '착' 소리. 셰이커보다 밝고 짧다."""
    t = _t(0.1, sr)
    grains = filt(_noise(len(t), 12), sr, "bandpass", [4500, 12000])
    env = np.where(t < 0.012, t / 0.012, np.exp(-(t - 0.012) * 55))
    return _norm(grains * env)


MEMBER_SAMPLES = {"drum": tom, "tambourine": tambourine, "shaker": shaker, "woodblock": woodblock, "maracas": maracas}


# ---------- 음 있는 악기 ----------
@lru_cache(maxsize=1024)
def pluck(midi: int, dur: float, sr: int, decay: float, bright: float, seed: int = 0) -> np.ndarray:
    """카플러스-스트롱 방식으로 줄을 튕기는 소리를 만든다."""
    f = midi_hz(midi)
    n = max(2, int(round(sr / f - 0.5)))
    length = int(dur * sr)
    exc = _noise(n, 100 + seed)
    exc = signal.lfilter([bright, 1.0 - bright], [1.0], exc)
    x = np.zeros(length)
    x[: min(n, length)] = exc[: min(n, length)]
    a = np.zeros(n + 2)
    a[0] = 1.0
    a[n] = -0.5 * decay
    a[n + 1] = -0.5 * decay
    y = signal.lfilter([1.0], a, x)
    return _norm(_tail_fade(y, sr), 0.9)


@lru_cache(maxsize=1024)
def steel(midi: int, dur: float, sr: int) -> np.ndarray:
    f = midi_hz(midi)
    t = _t(dur, sr)
    parts = [(1.0, 1.0, 3.2), (2.0, 0.55, 4.5), (3.0, 0.22, 6.0), (4.02, 0.12, 8.0), (5.0, 0.06, 10.0)]
    y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in parts)
    ping = np.sin(2 * np.pi * f * t + 1.5 * np.exp(-t * 30) * np.sin(2 * np.pi * f * 2.4 * t)) * np.exp(-t * 25) * 0.4
    return _norm(_tail_fade((y + ping) * (1 - np.exp(-t * 800)), sr), 0.9)


@lru_cache(maxsize=1024)
def bell(midi: int, dur: float, sr: int) -> np.ndarray:
    f = midi_hz(midi)
    t = _t(dur, sr)
    parts = [(1.0, 1.0, 2.0), (2.756, 0.5, 3.8), (5.404, 0.28, 6.5), (8.933, 0.12, 9.0)]
    y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in parts)
    click = filt(_noise(len(t), 9), sr, "highpass", 3000) * np.exp(-t * 500) * 0.05
    return _norm(_tail_fade((y + click) * (1 - np.exp(-t * 2000)), sr), 0.9)


@lru_cache(maxsize=1024)
def xylo(midi: int, dur: float, sr: int) -> np.ndarray:
    """실로폰: 나무 건반을 채로 친 소리 (짧게 사라지는 배음)."""
    f = midi_hz(midi)
    t = _t(dur, sr)
    parts = [(1.0, 1.0, 9.0), (3.93, 0.45, 22.0), (9.8, 0.12, 40.0)]
    y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in parts)
    knock = filt(_noise(len(t), 10), sr, "bandpass", [800, 3000]) * np.exp(-t * 300) * 0.15
    return _norm(_tail_fade((y + knock) * (1 - np.exp(-t * 3000)), sr), 0.9)


@lru_cache(maxsize=512)
def bass_note(midi: int, dur: float, sr: int) -> np.ndarray:
    f = midi_hz(midi)
    t = _t(dur, sr)
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    env = (1 - np.exp(-t * 300)) * (0.6 + 0.4 * np.exp(-t * 6))
    return _norm(_tail_fade(filt(y * env, sr, "lowpass", 900), sr, 0.04), 0.9)


def inst_note(inst: str, midi: int, sr: int, seed: int = 0) -> np.ndarray:
    """계절 악기 한 음."""
    if inst == "ukulele":
        return pluck(midi, 0.8, sr, 0.993, 0.65, seed % 3)
    if inst == "acoustic_guitar":
        return pluck(midi, 1.4, sr, 0.996, 0.45, seed % 3)
    if inst == "steel_drum":
        return steel(midi, 0.9, sr)
    return bell(midi, 1.5, sr)


@lru_cache(maxsize=1024)
def piano(midi: int, dur: float, sr: int) -> np.ndarray:
    """피아노: 망치가 줄을 친 소리. 높은 배음일수록 빨리 사라지고 조금씩 높아진다 (줄의 뻣뻣함)."""
    f = midi_hz(midi)
    t = _t(dur, sr)
    y = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > sr / 2 - 500:
            break
        y += (1.0 / k ** 1.1) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.6 + 0.9 * k))
    hammer = filt(_noise(len(t), 13), sr, "bandpass", [800, 5000]) * np.exp(-t * 300) * 0.08
    return _norm(_tail_fade((y + hammer) * (1 - np.exp(-t * 1500)), sr), 0.9)
