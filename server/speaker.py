"""목소리 특징(화자 임베딩)과 말소리 구간 찾기(VAD).

sherpa-onnx 모델(CAM++ 화자 모델, Silero VAD)이 있으면 그걸 쓰고,
없으면 소리 크기·주파수 모양으로 대신한다 (정확도는 떨어짐).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SHERPA_DIR, SPEAKER_MODEL, SR_IN, VAD_MODEL


@dataclass
class Segment:
    start: int  # 녹음 시작부터 샘플 위치
    samples: np.ndarray

    @property
    def start_sec(self) -> float:
        return self.start / SR_IN

    @property
    def end_sec(self) -> float:
        return (self.start + len(self.samples)) / SR_IN


def _db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9))


# ---------- 화자 임베딩 ----------
def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 60.0, fmax: float = 7600.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        for k in range(a, b):
            fb[i - 1, k] = (k - a) / max(b - a, 1)
        for k in range(b, c):
            fb[i - 1, k] = (c - k) / max(c - b, 1)
    return fb


_FB = _mel_filterbank(SR_IN, 512, 32)


def fallback_embedding(x: np.ndarray) -> np.ndarray:
    """모델이 없을 때 쓰는 대략적인 목소리 특징: 로그 멜 스펙트럼의 평균과 흩어짐."""
    frame, hop = 400, 160
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(frame)[None, :]
    spec = np.abs(np.fft.rfft(frames, 512)) ** 2
    mel = np.log(spec @ _FB.T + 1e-8)
    energy = mel.mean(axis=1)
    voiced = mel[energy > np.percentile(energy, 40)] if n > 4 else mel
    feat = np.concatenate([voiced.mean(axis=0), voiced.std(axis=0)])
    return (feat - feat.mean()).astype(np.float32)


class SpeakerEncoder:
    def __init__(self, models_dir: Path = SHERPA_DIR, threads: int = 2, log=print):
        self.kind, self.name, self.dim = "fallback", "mel-stats-v1", 64
        self._ext = None
        self._lock = threading.Lock()
        path = Path(models_dir) / SPEAKER_MODEL
        try:
            import sherpa_onnx

            if path.exists():
                cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=threads,
                                                                  debug=False, provider="cpu")
                if cfg.validate():
                    self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
                    self.kind, self.name, self.dim = "sherpa", "campplus-zh-en-advanced", int(self._ext.dim)
            else:
                log(f"[알림] 화자 모델이 없어서 대신 쓰는 방법으로 해: {path} (python -m server --download)")
        except ImportError:
            log("[알림] sherpa-onnx가 없어서 대신 쓰는 화자 구분 방법으로 해 (pip install sherpa-onnx)")

    def embed(self, x: np.ndarray) -> np.ndarray | None:
        """L2 정규화한 목소리 특징. 0.3초보다 짧으면 None."""
        if x is None or len(x) < int(0.3 * SR_IN):
            return None
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._ext is not None:
            with self._lock:
                s = self._ext.create_stream()
                s.accept_waveform(SR_IN, x)
                s.input_finished()
                e = np.asarray(self._ext.compute(s), dtype=np.float32)
        else:
            e = fallback_embedding(x)
        norm = float(np.linalg.norm(e))
        return e / norm if norm > 0 else None


# ---------- 말소리 구간 ----------
class EnergyVad:
    """모델이 없을 때 쓰는 소리 크기 기준 구간 찾기."""

    frame = 320  # 20ms

    def __init__(self, min_silence: float = 0.3, min_speech: float = 0.15, max_speech: float = 8.0):
        self.min_silence = int(min_silence * SR_IN / self.frame)
        self.min_speech = int(min_speech * SR_IN / self.frame)
        self.max_speech = int(max_speech * SR_IN / self.frame)
        self.buf = np.zeros(0, np.float32)
        self.pos = 0  # buf[0]의 녹음 내 위치
        self.history: list[float] = []
        self.in_speech = False
        self.seg_start = 0
        self.silence = 0
        self.frames: list[np.ndarray] = []

    def _floor(self) -> float:
        h = self.history[-150:]
        return float(np.percentile(h, 15)) if len(h) >= 10 else -60.0

    def accept(self, x: np.ndarray) -> list[Segment]:
        out: list[Segment] = []
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        while len(self.buf) >= self.frame:
            f, self.buf = self.buf[: self.frame], self.buf[self.frame:]
            db = _db(f)
            self.history.append(db)
            voiced = db > max(-48.0, self._floor() + 9.0)
            if voiced:
                if not self.in_speech:
                    self.in_speech, self.seg_start, self.frames = True, self.pos, []
                self.silence = 0
                self.frames.append(f)
            elif self.in_speech:
                self.silence += 1
                self.frames.append(f)
                if self.silence >= self.min_silence:
                    out += self._close(trim=self.silence)
            if self.in_speech and len(self.frames) >= self.max_speech:
                out += self._close(trim=0)
            self.pos += self.frame
        return out

    def _close(self, trim: int) -> list[Segment]:
        frames = self.frames[: len(self.frames) - trim] if trim else self.frames
        start = self.seg_start
        self.in_speech, self.frames, self.silence = False, [], 0
        if len(frames) < self.min_speech:
            return []
        return [Segment(start, np.concatenate(frames))]

    def flush(self) -> list[Segment]:
        return self._close(trim=self.silence) if self.in_speech else []

    def speech_now(self) -> bool:
        return self.in_speech


class StreamVad:
    """녹음 조각을 계속 넣으면 끝난 말소리 구간을 돌려준다."""

    def __init__(self, models_dir: Path = SHERPA_DIR, threshold: float = 0.4, min_silence: float = 0.3,
                 min_speech: float = 0.15, max_speech: float = 8.0):
        self._vad = None
        self.window = 512
        self._buf = np.zeros(0, np.float32)
        path = Path(models_dir) / VAD_MODEL
        try:
            import sherpa_onnx

            if path.exists():
                cfg = sherpa_onnx.VadModelConfig()
                cfg.silero_vad.model = str(path)
                cfg.silero_vad.threshold = threshold
                cfg.silero_vad.min_silence_duration = min_silence
                cfg.silero_vad.min_speech_duration = min_speech
                cfg.silero_vad.max_speech_duration = max_speech
                cfg.silero_vad.window_size = self.window
                cfg.sample_rate = SR_IN
                cfg.num_threads = 1
                if cfg.validate():
                    self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=600)
        except ImportError:
            pass
        self.kind = "silero" if self._vad is not None else "energy"
        self._energy = None if self._vad is not None else EnergyVad(min_silence, min_speech, max_speech)

    def _drain(self) -> list[Segment]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append(Segment(int(seg.start), np.asarray(seg.samples, dtype=np.float32)))
            self._vad.pop()
        return out

    def accept(self, x: np.ndarray) -> list[Segment]:
        if self._energy is not None:
            return self._energy.accept(x)
        self._buf = np.concatenate([self._buf, x.astype(np.float32)])
        n = (len(self._buf) // self.window) * self.window
        for i in range(0, n, self.window):
            self._vad.accept_waveform(self._buf[i:i + self.window])
        self._buf = self._buf[n:]
        return self._drain()

    def flush(self) -> list[Segment]:
        if self._energy is not None:
            return self._energy.flush()
        if len(self._buf):
            pad = np.zeros(self.window - len(self._buf), np.float32)
            self._vad.accept_waveform(np.concatenate([self._buf, pad]))
            self._buf = np.zeros(0, np.float32)
        self._vad.flush()
        return self._drain()

    def speech_now(self) -> bool:
        if self._energy is not None:
            return self._energy.speech_now()
        return bool(self._vad.is_speech_detected())


def voiced_part(x: np.ndarray, min_db: float = -45.0) -> np.ndarray:
    """조각 안에서 소리가 나는 20ms 칸만 이어 붙인다 (지금 말하는 사람 찾기용)."""
    frame = 320
    n = len(x) // frame
    if n == 0:
        return np.zeros(0, np.float32)
    frames = x[: n * frame].reshape(n, frame)
    dbs = np.array([_db(f) for f in frames])
    keep = dbs > max(min_db, float(np.max(dbs)) - 25.0)
    return frames[keep].reshape(-1) if keep.any() else np.zeros(0, np.float32)


def speech_report(x: np.ndarray, models_dir: Path = SHERPA_DIR) -> dict:
    """등록 녹음 품질: 말한 시간, 말소리 크기, 잡음 대비."""
    vad = StreamVad(models_dir, min_silence=0.25, min_speech=0.1)
    segs = vad.accept(x) + vad.flush()
    mask = np.zeros(len(x), bool)
    for s in segs:
        mask[s.start:s.start + len(s.samples)] = True
    speech = x[mask]
    noise = x[~mask]
    speech_sec = float(mask.sum() / SR_IN)
    level = _db(speech)
    noise_db = _db(noise) if noise.size > SR_IN // 10 else -90.0
    return {"speechSec": round(speech_sec, 2), "levelDb": round(level, 1), "snrDb": round(level - noise_db, 1),
            "speech": speech, "vad": vad.kind}
