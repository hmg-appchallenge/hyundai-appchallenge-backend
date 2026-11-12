"""TTS (Supertonic-3, CPU에서 ONNX로 동작, 한국어 지원, 키 필요 없음)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dsp import SR, filt, normalize_rms, pitch_shift, resample, trim_silence


class SupertonicTTS:
    def __init__(self, voice: str = "F1", models_dir: Path | None = None, steps: int = 16):
        from supertonic import TTS

        mdir = Path(models_dir) / "supertonic3" if models_dir else None
        self.engine = TTS(model="supertonic-3", model_dir=mdir, auto_download=True)
        self.voice_name = voice
        self.style = self.engine.get_voice_style(voice)
        self.steps = steps
        self.sample_rate = int(self.engine.sample_rate)
        self.name = f"supertonic-3/{voice}"

    def voices(self) -> list[str]:
        return list(getattr(self.engine, "voice_style_names", [])) or [f"F{i}" for i in range(1, 6)] + [f"M{i}" for i in range(1, 6)]

    def use_voice(self, voice: str) -> None:
        self.style = self.engine.get_voice_style(voice)
        self.voice_name = voice
        self.name = f"supertonic-3/{voice}"

    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        speed = float(min(2.0, max(0.7, speed)))
        wav, _dur = self.engine.synthesize(text, voice_style=self.style, total_steps=self.steps,
                                           speed=speed, lang="ko", silence_duration=0.1)
        return np.asarray(wav, dtype=np.float32).reshape(-1)


class ToneTTS:
    """시험용 가짜 TTS: 음절마다 짧은 모음 같은 소리를 낸다."""

    sample_rate = 24000
    name = "tone-mock"

    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        sr = self.sample_rate
        syl = max(1, sum(1 for ch in text if "가" <= ch <= "힣"))
        dur = 0.17 / max(speed, 0.1)
        t = np.arange(int(dur * sr)) / sr
        out = [np.zeros(int(0.05 * sr), np.float32)]
        for i in range(syl):
            f = 220 * (1.0 + 0.08 * ((i * 7) % 5))
            tone = sum((1 / k) * np.sin(2 * np.pi * f * k * t) for k in range(1, 6))
            env = np.minimum(1, t * 60) * np.minimum(1, (dur - t) * 40)
            out.append((tone * env * 0.3).astype(np.float32))
        out.append(np.zeros(int(0.05 * sr), np.float32))
        return np.concatenate(out)


class Voice:
    """TTS 결과를 곡에 맞게 다듬는다: 아이 같은 음높이, 길이 맞추기, 정리."""

    def __init__(self, tts, pitch: float = 3.0, sr: int = SR):
        self.tts = tts
        self.pitch = pitch
        self.sr = sr
        self.cache: dict[tuple[str, float], np.ndarray] = {}

    def _render(self, text: str, speed: float) -> np.ndarray:
        """같은 말, 같은 속도는 한 번만 합성한다 (속도는 0.05 단위로 맞춤)."""
        speed = round(round(speed / 0.05) * 0.05, 2)
        key = (text, speed)
        if key not in self.cache:
            raw = self.tts.synth(text, speed=speed)
            y = resample(raw, self.tts.sample_rate, self.sr)
            y = trim_silence(y, self.sr, -40.0, 6.0)
            if self.pitch:
                y = pitch_shift(y, self.sr, self.pitch)
            y = filt(y, self.sr, "highpass", 110)
            self.cache[key] = normalize_rms(y, -16.0, 0.9)
        return self.cache[key]

    def say(self, text: str, max_dur: float | None = None, fill_dur: float | None = None) -> np.ndarray:
        """max_dur보다 길면 빠르게, fill_dur에 비해 너무 짧으면 조금 느리게 다시 만든다."""
        y = self._render(text, 1.0)
        natural = len(y) / self.sr
        if max_dur and natural > max_dur:
            y = self._render(text, min(2.0, natural / max_dur * 1.04))
        elif fill_dur and natural < 0.55 * fill_dur:
            speed = max(0.8, natural / (0.7 * fill_dur))
            if speed < 0.97:
                y = self._render(text, speed)
        return y
