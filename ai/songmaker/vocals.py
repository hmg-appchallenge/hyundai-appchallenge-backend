"""보컬 트랙 세 가지.

A: 아이 목소리만 - 아이가 말한 의성어와 단어를 잘라 박자와 화음에 맞춰 배치
B: TTS만 - TTS가 가사를 박자에 맞춰 챈트로 부름 (아이 목소리 없음)
C: 섞기 - TTS가 가사를 부르고, 아이 목소리가 의성어와 단어를 메아리처럼 받음
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arrange import CHORUS_BARS, VERSE_LINE_BARS, Song
from .dsp import add_at, envelope, fade, filt, hz_to_midi, normalize_rms, pitch_shift, yin_f0
from .lyrics import Lyrics, syllables
from .tts import Voice

VARIANTS = {
    "A": ("child", "아이 목소리만"),
    "B": ("tts", "TTS만"),
    "C": ("mix", "섞기 (TTS 가사 + 아이 목소리)"),
}


@dataclass
class Clip:
    audio: np.ndarray
    onset: float
    f0: float | None
    text: str
    member: str


def extract_clip(audio: np.ndarray, sr: int, start: float, end: float, frac_start: float = 0.0,
                 frac_end: float = 1.0, prev_end: float | None = None, next_start: float | None = None,
                 text: str = "", member: str = "") -> Clip | None:
    """인식 시간 근처에서 소리 에너지를 보고 단어 한 덩어리를 깔끔하게 잘라 낸다."""
    total = len(audio) / sr
    dur = max(end - start, 0.05)
    s = start + dur * frac_start
    e = start + dur * frac_end
    lo = max(0.0, s - (0.04 if frac_start > 0.02 else 0.12))
    hi = min(total, e + (0.06 if frac_end < 0.98 else 0.25))
    if prev_end is not None and frac_start <= 0.02:
        lo = max(lo, prev_end - 0.02)
    if next_start is not None and frac_end >= 0.98:
        hi = min(hi, next_start + 0.02)
    if hi - lo < 0.06:
        return None
    seg = audio[int(lo * sr):int(hi * sr)]
    env = envelope(seg, sr, 10.0)
    mx = float(env.max()) if env.size else 0.0
    if mx < 1e-4:
        return None
    a = int(np.where(env > 0.12 * mx)[0][0])
    if frac_end < 0.98:
        pred = int((e - lo) * sr)
        w = int(0.06 * sr)
        i0, i1 = max(a + int(0.05 * sr), pred - w), min(len(env) - 1, pred + w)
        b = int(i0 + np.argmin(env[i0:i1])) if i1 > i0 else pred
    else:
        b = int(np.where(env > 0.05 * mx)[0][-1])
    a2 = max(0, a - int(0.012 * sr))
    b2 = min(len(seg), b + int(0.05 * sr))
    clip = seg[a2:b2]
    if len(clip) < int(0.06 * sr):
        return None
    clip = filt(clip, sr, "highpass", 90)
    clip = fade(clip, sr, 0.004, 0.03)
    clip = normalize_rms(clip, -16.0, 0.9)
    f = yin_f0(clip, sr, fmin=110.0, fmax=900.0)
    f0 = float(np.median(f)) if f.size >= 3 else None
    return Clip(clip, (a - a2) / sr, f0, text, member)


class Placer:
    def __init__(self, song: Song):
        self.song = song
        self.sr = song.sr
        self.buf = np.zeros(song.n, np.float32)
        self._tune_cache: dict[tuple[int, int], np.ndarray] = {}
        self.log: list[dict] = []

    def at(self, bar: int, beat: float) -> float:
        bar += int(beat // 4)
        return self.song.t(bar, beat % 4)

    def tuned(self, clip: Clip, target: int | None, bar: int) -> np.ndarray:
        """아이 목소리를 목표 음으로 옮긴다. 4반음보다 멀면 그 마디 화음 중 가까운 음으로."""
        if target is None or clip.f0 is None:
            return clip.audio
        cur = hz_to_midi(clip.f0)
        best = min((target + 12 * k for k in range(-3, 4)), key=lambda m: abs(m - cur))
        if abs(best - cur) > 4.0:
            tones = self.song.chord_tones(min(bar, self.song.bars - 1))
            best = min((t + 12 * k for t in tones for k in range(-3, 4)), key=lambda m: abs(m - cur))
        key = (id(clip), best)
        if key not in self._tune_cache:
            shift = float(np.clip(best - cur, -5.0, 5.0))
            self._tune_cache[key] = pitch_shift(clip.audio, self.sr, shift)
        return self._tune_cache[key]

    def put(self, audio: np.ndarray, t: float, onset: float = 0.008, gain: float = 1.0, what: str = "") -> float:
        add_at(self.buf, audio, int(round((t - onset) * self.sr)), gain)
        end = t - onset + len(audio) / self.sr
        if what:
            self.log.append({"t": round(t, 3), "what": what})
        return end

    def clip(self, clip: Clip | None, bar: int, beat: float, target: int | None = None, gain: float = 0.9) -> bool:
        if clip is None:
            return False
        self.put(self.tuned(clip, target, bar + int(beat // 4)), self.at(bar, beat), clip.onset, gain, f"아이:{clip.text}")
        return True

    def say(self, voice: Voice, text: str, bar: int, beat: float, max_beats: float = 1.0, gain: float = 1.0) -> float:
        slot = max_beats * self.song.spb
        audio = voice.say(text, slot * 0.95, slot)
        return self.put(audio, self.at(bar, beat), 0.008, gain, f"TTS:{text}")

    def chant(self, voice: Voice, line: str, bar: int, beat0: float = 0.0, max_eighths: int = 14,
              gain: float = 1.0) -> float:
        """어절마다 음절 수만큼 8분음표 자리를 주고, 어절 첫소리를 박자에 맞춘다. 끝난 박 위치를 돌려준다."""
        chunks = line.replace(",", " ").split()
        syl = [max(1, syllables(c)) for c in chunks]
        total = sum(syl)
        unit = 0.5 if total <= max_eighths else 0.5 * max_eighths / total
        pos = beat0
        for chunk, n in zip(chunks, syl):
            slot = n * unit
            self.say(voice, chunk, bar, pos, slot, gain)
            pos += slot
        return pos


def _targets(song: Song, bar: int, idx: int) -> int:
    tones = song.chord_tones(min(bar, song.bars - 1))
    return tones[idx % len(tones)] + 12


def render(variant: str, song: Song, lyr: Lyrics, noun_clips: list[Clip | None],
           ono_clips: dict[str, Clip | None], ono_texts: dict[str, str], voice: Voice | None) -> tuple[np.ndarray, list[dict]]:
    p = Placer(song)
    main, mel, extra = ono_clips.get("main"), ono_clips.get("melody"), ono_clips.get("extra")
    main_t = ono_texts.get("main", "둥둥")
    mel_t = ono_texts.get("melody", "랄랄라")
    nclips = [c for c in noun_clips if c is not None]
    words = lyr.words

    def noun_clip(i: int) -> Clip | None:
        return noun_clips[i] if i < len(noun_clips) and noun_clips[i] is not None else (nclips[i % len(nclips)] if nclips else None)

    def child_or_tts(clip: Clip | None, text: str, bar: int, beat: float, target: int | None, gain: float = 0.9) -> None:
        if not p.clip(clip, bar, beat, target, gain) and voice is not None:
            p.say(voice, text, bar, beat, 1.0, gain)

    if variant == "A":
        p.clip(main, 1, 0, _targets(song, 1, 0))
        p.clip(main, 1, 2, _targets(song, 1, 2))
        p.clip(mel or extra, 1, 3, _targets(song, 1, 1))
        for i, bar in enumerate(VERSE_LINE_BARS):
            nc = noun_clip(i)
            p.clip(nc, bar, 0, _targets(song, bar, 0))
            p.clip(main, bar, 1, _targets(song, bar, 2), 0.75)
            p.clip(nc, bar, 2, _targets(song, bar, 2))
            p.clip(extra or main, bar, 3, _targets(song, bar, 1), 0.75)
            p.clip(nc, bar + 1, 0, _targets(song, bar + 1, 1))
            p.clip(mel or main, bar + 1, 2, _targets(song, bar + 1, 0), 0.85)
        for k, bar in enumerate(CHORUS_BARS):
            mel_notes = song.hook_notes(bar)
            p.clip(main, bar, 0, mel_notes[0])
            p.clip(noun_clip(k), bar, 1, mel_notes[1])
            p.clip(main, bar, 2, mel_notes[2])
            p.clip(mel or extra, bar, 3, mel_notes[3])
        outro = song.hook_notes(14)
        p.clip(noun_clip(0), 14, 0, outro[0])
        p.clip(mel or main, 14, 2, outro[1])
        p.clip(main, 15, 0, song.melody_base())
        p.clip(mel or main, 16, 0, song.melody_base())
    elif variant == "B":
        if voice is None:
            raise RuntimeError("TTS가 없어서 B 버전을 만들 수 없어")
        p.say(voice, main_t, 1, 0)
        p.say(voice, main_t, 1, 2)
        p.say(voice, mel_t, 1, 3)
        for i, bar in enumerate(VERSE_LINE_BARS):
            end = p.chant(voice, lyr.verse[i], bar, 0, 14)
            if i < len(words) and end <= 6.0:
                p.say(voice, words[i], bar, max(6.0, math.ceil(end)), 1.0, 0.85)
        for k, bar in enumerate(CHORUS_BARS):
            p.chant(voice, lyr.hook[k % 2], bar, 0, 8)
        p.chant(voice, lyr.title, 14, 0, 4)
        p.say(voice, mel_t, 15, 2, 1.5)
    else:  # C
        child_or_tts(main, main_t, 1, 0, _targets(song, 1, 0))
        child_or_tts(main, main_t, 1, 2, _targets(song, 1, 2))
        child_or_tts(mel or extra, mel_t, 1, 3, _targets(song, 1, 1))
        for i, bar in enumerate(VERSE_LINE_BARS):
            end = p.chant(voice, lyr.verse[i], bar, 0, 14) if voice else 0.0
            echo_beat = max(6.0, math.ceil(end))
            if echo_beat <= 6.5:
                p.clip(noun_clip(i), bar, echo_beat, _targets(song, bar + 1, 0))
            p.clip(main if i % 2 == 0 else (extra or main), bar + 1, 3, _targets(song, bar + 1, 2), 0.8)
        for k, bar in enumerate(CHORUS_BARS):
            if k % 2 == 0:
                child_or_tts(main, main_t, bar, 0, song.hook_notes(bar)[0])
                child_or_tts(main, main_t, bar, 1, song.hook_notes(bar)[1])
            else:
                child_or_tts(mel or main, mel_t, bar, 0, song.hook_notes(bar)[0])
            if voice:
                p.chant(voice, lyr.hook_tail[k % 2], bar, 2, 4)
        child_or_tts(noun_clip(0), words[0] if words else main_t, 14, 0, song.hook_notes(14)[0])
        child_or_tts(mel or main, mel_t, 15, 2, song.melody_base())
        p.clip(main, 16, 0, song.melody_base(), 0.8)
    return p.buf, p.log
