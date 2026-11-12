"""올라온 녹음을 16kHz 모노 float32로 바꾼다."""
from __future__ import annotations

import io

import numpy as np

from .config import SR_IN


class BadAudio(Exception):
    pass


def pcm16_to_float(data: bytes) -> np.ndarray:
    n = len(data) // 2
    return np.frombuffer(data[: n * 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    try:
        import soxr

        return soxr.resample(x, sr, target).astype(np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target))
        return resample_poly(x, target // g, sr // g).astype(np.float32)


def decode(data: bytes, target_sr: int = SR_IN) -> np.ndarray:
    """wav·flac·ogg·mp3·m4a는 파일로 읽고, 머리글이 없으면 PCM 16비트 16kHz 모노로 본다."""
    if not data:
        raise BadAudio("빈 파일")
    head = data[:12]
    is_container = head[:4] in (b"RIFF", b"fLaC", b"OggS") or head[:3] == b"ID3" or head[4:8] == b"ftyp" \
        or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)
    if not is_container:
        return pcm16_to_float(data) if target_sr == SR_IN else _resample(pcm16_to_float(data), SR_IN, target_sr)
    try:
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return _resample(x.mean(axis=1), sr, target_sr)
    except Exception:
        pass
    try:
        import av  # faster-whisper와 함께 설치된다

        with av.open(io.BytesIO(data)) as box:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
            out = []
            for frame in box.decode(audio=0):
                for f in resampler.resample(frame):
                    out.append(f.to_ndarray().reshape(-1))
            if not out:
                raise BadAudio("소리가 없음")
            return np.concatenate(out).astype(np.float32) / 32768.0
    except BadAudio:
        raise
    except Exception as e:
        raise BadAudio(f"읽을 수 없는 파일 ({type(e).__name__})") from e
