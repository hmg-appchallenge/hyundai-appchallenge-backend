"""녹음이 오기 전에 전체 흐름을 시험하려고, TTS로 아이 흉내 목소리 두 개를 만든다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dsp import SR
from .tts import Voice

SCRIPT = {
    "voice1": (["둥둥!", "둥둥!", "랄랄라~", "사과!", "강아지!", "빵빵!", "엄마!", "바다!"], "와 강아지다! 우리 바다 가요!"),
    "voice2": (["쿵쿵!", "랄랄라~", "고양이!", "기차!", "할머니!", "공원!"], "고양이랑 기차 타고 공원 가요!"),
}
FAKE_VOICES = {"voice1": ("F2", 4.0), "voice2": ("F4", 3.0)}
PREVIEW_LINE = "사과가 데굴데굴 굴러가요"


def make_fake_inputs(tts, out_dir: Path) -> list[Path]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    original = getattr(tts, "voice_name", None)
    available = set(tts.voices()) if hasattr(tts, "voices") else set()
    paths = []
    for vid, (words, free) in SCRIPT.items():
        name, pitch = FAKE_VOICES[vid]
        if hasattr(tts, "use_voice") and (not available or name in available):
            tts.use_voice(name)
        v = Voice(tts, pitch=pitch)
        gap = np.zeros(int(0.5 * SR), np.float32)
        parts = [np.zeros(int(0.4 * SR), np.float32)]
        for w in words:
            parts += [v.say(w) * 0.8, gap]
        parts += [v.say(free) * 0.8, gap]
        p = out_dir / f"{vid}.wav"
        sf.write(str(p), np.concatenate(parts), SR, subtype="PCM_16")
        paths.append(p)
    if original and hasattr(tts, "use_voice"):
        tts.use_voice(original)
    return paths


def voices_preview(tts, out_path: Path, pitch: float, log) -> list[str]:
    """목소리 10개가 같은 가사를 차례로 읽는 파일. 귀여운 목소리를 고르는 용도."""
    from .mix import save_mp3

    names = tts.voices()
    original = getattr(tts, "voice_name", None)
    parts = []
    for name in names:
        tts.use_voice(name)
        v = Voice(tts, pitch=pitch)
        parts += [v.say(PREVIEW_LINE), np.zeros(int(0.8 * SR), np.float32)]
        log(f"  {len(parts) // 2:>2}번째: {name}")
    if original:
        tts.use_voice(original)
    mono = np.concatenate(parts) * 0.8
    save_mp3(np.stack([mono, mono], axis=1), SR, out_path)
    return names
