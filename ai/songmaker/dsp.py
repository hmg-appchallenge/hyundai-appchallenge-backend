"""오디오 처리 도구 모음. numpy와 scipy만 쓴다."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

SR = 44100


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def normalize_rms(x: np.ndarray, target_db: float = -18.0, peak_cap: float = 0.9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r < 1e-6:
        return x
    y = x * (db_to_amp(target_db) / r)
    p = peak(y)
    if p > peak_cap:
        y = y * (peak_cap / p)
    return y.astype(np.float32)


def normalize_peak(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    p = peak(x)
    return (x * (target / p)).astype(np.float32) if p > 0 else x


def fade(x: np.ndarray, sr: int, fade_in: float = 0.005, fade_out: float = 0.02) -> np.ndarray:
    y = np.array(x, dtype=np.float32, copy=True)
    n_in = min(len(y), int(fade_in * sr))
    n_out = min(len(y), int(fade_out * sr))
    if n_in > 1:
        y[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 1:
        y[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return y


def filt(x: np.ndarray, sr: int, kind: str, freq, order: int = 2) -> np.ndarray:
    """kind: lowpass, highpass, bandpass. freq는 Hz (bandpass는 [낮은, 높은])."""
    if np.size(x) == 0:
        return np.asarray(x, dtype=np.float32)
    sos = signal.butter(order, freq, btype=kind, fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def resample(x: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or abs(sr_in - sr_out) < 1e-6:
        return x
    try:
        import soxr

        return soxr.resample(x, float(sr_in), float(sr_out)).astype(np.float32)
    except ImportError:
        fr = Fraction(sr_out / sr_in).limit_denominator(2000)
        return signal.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def wsola(x: np.ndarray, rate: float, sr: int = SR, frame_ms: float = 30.0, tol_ms: float = 10.0) -> np.ndarray:
    """음높이는 그대로 두고 길이만 바꾼다. rate > 1이면 빨라져서 짧아진다."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or abs(rate - 1.0) < 1e-3:
        return x.copy()
    n = int(sr * frame_ms / 1000) // 2 * 2
    hs = n // 2
    tol = int(sr * tol_ms / 1000)
    win = signal.windows.hann(n, sym=False).astype(np.float32)
    ha = hs * rate
    lead = hs
    xp = np.concatenate([np.zeros(tol + lead, np.float32), x, np.zeros(tol + 2 * n, np.float32)])
    base = tol
    out_len = int(round(len(x) / rate))
    n_frames = (out_len + lead) // hs + 2
    y = np.zeros(n_frames * hs + n, np.float32)
    prev = -1
    for k in range(n_frames):
        nom = base + int(round(k * ha))
        if nom + n > len(xp):
            break
        if prev < 0:
            best = nom
        else:
            nat = prev + hs
            tmpl = xp[nat:nat + n]
            lo = max(nom - tol, 0)
            hi = min(nom + tol, len(xp) - n)
            if hi <= lo or len(tmpl) < n or not np.any(tmpl):
                best = nom
            else:
                seg = xp[lo:hi + n]
                c = signal.correlate(seg, tmpl, mode="valid", method="fft")
                best = lo + int(np.argmax(c))
        y[k * hs:k * hs + n] += xp[best:best + n] * win
        prev = best
    return y[lead:lead + out_len]


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """반음 단위로 음높이를 바꾼다. 길이는 그대로."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or abs(semitones) < 0.05:
        return x.copy()
    f = 2.0 ** (semitones / 12.0)
    y = resample(x, sr, sr / f)
    out = wsola(y, 1.0 / f, sr)
    if len(out) > len(x):
        out = out[:len(x)]
    elif len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)))
    return out.astype(np.float32)


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def yin_f0(x: np.ndarray, sr: int, fmin: float = 90.0, fmax: float = 800.0,
           hop: int = 256, threshold: float = 0.15) -> np.ndarray:
    """YIN 방식으로 프레임별 기본 주파수(Hz)를 구한다. 무성음 프레임은 뺀다."""
    x = np.asarray(x, dtype=np.float64)
    tau_min = max(2, int(sr / fmax))
    tau_max = int(sr / fmin)
    w = max(1024, tau_max)
    frame = w + tau_max
    if len(x) < frame:
        return np.array([])
    gate = 0.1 * peak(x)
    f0 = []
    taus = np.arange(tau_max + 1)
    for s in range(0, len(x) - frame, hop):
        fr = x[s:s + frame]
        if np.max(np.abs(fr[:w])) < gate:
            continue
        x0 = fr[:w]
        e = np.concatenate([[0.0], np.cumsum(fr * fr)])
        e0 = e[w]
        et = e[taus + w] - e[taus]
        r = signal.correlate(fr[:w + tau_max], x0, mode="valid", method="fft")
        d = e0 + et - 2.0 * r
        d[0] = 0.0
        cs = np.cumsum(d[1:])
        cmnd = np.ones(tau_max + 1)
        cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(cs, 1e-12)
        idx = np.where(cmnd[tau_min:] < threshold)[0]
        if idx.size == 0:
            continue
        t = int(idx[0] + tau_min)
        while t + 1 <= tau_max and cmnd[t + 1] < cmnd[t]:
            t += 1
        shift = 0.0
        if 1 <= t < tau_max:
            a, b, c = cmnd[t - 1], cmnd[t], cmnd[t + 1]
            den = a - 2.0 * b + c
            if abs(den) > 1e-12:
                shift = 0.5 * (a - c) / den
        f0.append(sr / (t + shift))
    return np.array(f0)


def envelope(x: np.ndarray, sr: int, win_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * win_ms / 1000))
    k = np.ones(n) / n
    return np.sqrt(np.maximum(np.convolve(np.square(x, dtype=np.float64), k, mode="same"), 0.0))


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -38.0, pad_ms: float = 8.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    env = envelope(x, sr, 5.0)
    th = env.max() * db_to_amp(thresh_db)
    idx = np.where(env > th)[0]
    if idx.size == 0:
        return x[:0]
    pad = int(sr * pad_ms / 1000)
    return x[max(0, idx[0] - pad):min(len(x), idx[-1] + pad)]


def compress(x: np.ndarray, sr: int, threshold_db: float = -20.0, ratio: float = 3.0,
             release_ms: float = 80.0, makeup_db: float = 0.0) -> np.ndarray:
    env = envelope(x, sr, 5.0)
    env_db = 20.0 * np.log10(env + 1e-9)
    gr = np.maximum(env_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    a = np.exp(-1.0 / (sr * release_ms / 1000.0))
    gr = np.maximum(gr, signal.lfilter([1 - a], [1, -a], gr))
    return (x * np.power(10.0, (makeup_db - gr) / 20.0)).astype(np.float32)


def reverb_ir(sr: int, seconds: float = 1.0, rt60: float = 0.8, seed: int = 0, lowpass: float = 6500.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    ir = filt(ir, sr, "lowpass", lowpass)
    ir[: int(0.015 * sr)] = 0.0
    ir = ir / np.sqrt(np.sum(ir.astype(np.float64) ** 2) + 1e-12)
    return ir.astype(np.float32)


def pan(x: np.ndarray, p: float) -> np.ndarray:
    """p: -1(왼쪽) ~ +1(오른쪽). 결과 모양은 (샘플 수, 2)."""
    th = (float(p) + 1.0) * np.pi / 4.0
    return np.stack([x * np.cos(th), x * np.sin(th)], axis=1).astype(np.float32)


def limit(st: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 3.0,
          release_ms: float = 80.0) -> np.ndarray:
    """스테레오 (n, 2) 신호의 피크를 ceiling 아래로 누른다."""
    thr = db_to_amp(ceiling_db)
    mono_peak = np.max(np.abs(st), axis=1)
    la = max(1, int(sr * lookahead_ms / 1000))
    pk = maximum_filter1d(mono_peak, size=2 * la + 1)
    red = 1.0 - np.minimum(1.0, thr / np.maximum(pk, 1e-9))
    a = np.exp(-1.0 / (sr * release_ms / 1000.0))
    red = np.maximum(red, signal.lfilter([1 - a], [1, -a], red))
    out = st * (1.0 - red)[:, None]
    return np.clip(out, -thr, thr).astype(np.float32)


def add_at(buf: np.ndarray, x: np.ndarray, start: int, gain: float = 1.0) -> None:
    """buf[start:]에 x를 더한다. 범위를 벗어난 부분은 자른다."""
    if x.size == 0 or start >= len(buf):
        return
    if start < 0:
        x = x[-start:]
        start = 0
    end = min(len(buf), start + len(x))
    if end > start:
        buf[start:end] += x[: end - start] * gain
