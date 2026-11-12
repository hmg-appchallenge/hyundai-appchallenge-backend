"""앱에 넣을 소리 파일을 노래 엔진과 같은 소리로 만든다.

python -m songmaker.app_sounds <폴더>
- beat_080_2.wav, beat_095_4.wav, beat_110_8.wav, beat_125_16.wav : S03~S05에서 계속 도는 2마디 비트 (끊김 없이 반복)
- inst_<악기>_preview.wav : 멤버 추가 팝업의 "누르면 3초 미리 듣기"
- inst_<악기>_hit.wav : S04 "소리 낸 사람의 악기가 울려요"
모두 22.05kHz 모노 16비트 (앱 용량을 줄이려고).
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from . import synth
from .arrange import _bass, _drums
from .beat import BeatSpec, DEFAULT_HOOK, DEFAULT_PROG
from .dsp import add_at, db_to_amp, filt, limit, normalize_rms
from .session_song import MEMBER_PATTERNS, PERCUSSION, PITCHED_REGISTER, FormSong, drums_for, member_hit, member_note

SR = 44100  # 엔진과 같은 샘플레이트로 만들고
OUT_SR = 22050  # 저장할 때 줄인다 (높은 소리가 깨지지 않게)
BANDS = [(80, 2), (95, 4), (110, 8), (125, 16)]


def _write(path: Path, x: np.ndarray) -> None:
    from .dsp import resample

    pcm = (np.clip(resample(x, SR, OUT_SR), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_SR)
        w.writeframes(pcm.tobytes())


def _spec(bpm: int, beat_type: int) -> BeatSpec:
    prog = {k: list(v) for k, v in DEFAULT_PROG.items()}
    prog["verse"] = ["I", "V"]
    return BeatSpec(bpm=bpm, swing=0.0, key="C", instrument_set="acoustic_guitar", instrument_style="arp8",
                    progression=prog, drums=drums_for(beat_type), fills=[], members={},
                    hook=[list(h) for h in DEFAULT_HOOK])


def beat_loop(bpm: int, beat_type: int) -> np.ndarray:
    """같은 마디를 5개 연주하고 가운데 2마디(3·4번째)만 잘라 낸다.
    3번째 마디 첫머리에는 앞 마디(4번째와 같은 마디)의 울림이 이미 들어 있어서, 반복해도 이음매가 들리지 않는다."""
    song = FormSong(_spec(bpm, beat_type), form=[("verse", 5)], sr=SR, tail=1.0)
    song.crash_bars = ()
    names = ["kick", "snare", "hat", "clap", "crash", "bass", "inst", "melody"]
    st = {n: np.zeros(song.n, np.float32) for n in names}
    _drums(song, st)
    _bass(song, st)
    levels = {"kick": -6.0, "snare": -3.0, "hat": -9.0, "clap": -4.0, "bass": -10.0}
    mix = sum(st[k] * db_to_amp(db) for k, db in levels.items())
    bar = 4 * song.spb * SR
    body = mix[int(round(2 * bar)):int(round(4 * bar))].copy()
    body = filt(body, SR, "highpass", 35)
    body = normalize_rms(body, -17.0, 0.95)
    return limit(np.stack([body, body], axis=1), SR, -1.5, 3.0, 60.0)[:, 0]


def preview(inst: str, bpm: int = 110) -> np.ndarray:
    """그 악기 패턴을 3초 동안 (후렴 패턴, 화음 I-V)."""
    song = FormSong(_spec(bpm, 8), form=[("chorus", 2)], sr=SR, tail=1.0)
    out = np.zeros(int(3.0 * SR), np.float32)
    pat = MEMBER_PATTERNS[inst]["chorus"]
    arp = 0
    for bar in range(2):
        tones = song.chord_tones(bar)
        for i, ch in enumerate(pat):
            if ch not in "xX":
                continue
            t = song.t(bar, i / 4.0)
            if t >= 2.7:
                continue
            if inst in PERCUSSION:
                smp = PERCUSSION[inst](SR)
            else:
                base = PITCHED_REGISTER[inst]
                tone = tones[arp % 3]
                while tone < base:
                    tone += 12
                while tone >= base + 12:
                    tone -= 12
                smp = member_note(inst, tone, SR)
                arp += 1
            add_at(out, smp, int(t * SR), 1.0 if ch == "X" else 0.75)
    n = int(0.25 * SR)
    out[-n:] *= np.linspace(1, 0, n)
    return normalize_rms(out, -18.0, 0.9)


def hit(inst: str) -> np.ndarray:
    x = member_hit(inst, SR)[: int(0.6 * SR)].copy()
    n = int(0.05 * SR)
    if len(x) > n:
        x[-n:] *= np.linspace(1, 0, n)
    return normalize_rms(x, -16.0, 0.9)


def render_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for bpm, bt in BANDS:
        p = out_dir / f"beat_{bpm:03d}_{bt}.wav"
        _write(p, beat_loop(bpm, bt))
        files.append(p)
    for inst in list(PERCUSSION) + list(PITCHED_REGISTER):
        for kind, fn in (("preview", preview), ("hit", hit)):
            p = out_dir / f"inst_{inst.lower()}_{kind}.wav"
            _write(p, fn(inst))
            files.append(p)
    return files


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("app_sounds")
    for f in render_all(target):
        print(f, f"{f.stat().st_size / 1024:.0f} KB")
