"""곡 구조(16마디)와 반주. 비트 악보(BeatSpec)를 받아 그대로 연주한다.

구조: 인트로 2마디 -> 벌스 8마디(가사 4줄, 줄마다 2마디) -> 후렴 4마디 -> 아웃트로 2마디
"""
from __future__ import annotations

import math

import numpy as np

from . import synth
from .beat import DEGREE_INFO, DRUM_PARTS, KEY_OFFSET, SCALE, BeatSpec
from .dsp import add_at

SECTIONS = ["intro"] * 2 + ["verse"] * 8 + ["chorus"] * 4 + ["outro"] * 2
SECTION_START = {"intro": 0, "verse": 2, "chorus": 10, "outro": 14}
VERSE_LINE_BARS = [2, 4, 6, 8]
CHORUS_BARS = [10, 11, 12, 13]
VELOCITY = {"x": 0.75, "X": 1.0}
PART_GAIN = {"kick": 1.0, "snare": 0.85, "hat": 0.5, "openHat": 0.45, "clap": 0.8}


class Song:
    def __init__(self, spec: BeatSpec, bars: int = 16, sr: int = 44100, tail: float = 2.5):
        self.spec = spec
        self.bars = bars
        self.sr = sr
        self.tail = tail
        self.bpm = spec.bpm
        self.instrument_set = spec.instrument_set

    @property
    def spb(self) -> float:
        return 60.0 / self.bpm

    def _swung(self, beat: float) -> float:
        """박 안의 위치를 스윙 곡선으로 옮긴다. 8분음표 뒷박(0.5)이 swing만큼 뒤로 밀린다."""
        whole = math.floor(beat + 1e-9)
        frac = beat - whole
        s = 0.5 + self.spec.swing / 6.0
        frac = frac * 2 * s if frac <= 0.5 else s + (frac - 0.5) * 2 * (1 - s)
        return whole + frac

    def t(self, bar: int, beat: float) -> float:
        return (bar * 4 + self._swung(beat)) * self.spb

    @property
    def length(self) -> float:
        return self.bars * 4 * self.spb + self.tail

    @property
    def n(self) -> int:
        return int(self.length * self.sr)

    def section(self, bar: int) -> str:
        return SECTIONS[bar] if 0 <= bar < len(SECTIONS) else "outro"

    def degree(self, bar: int) -> str:
        bar = min(max(bar, 0), self.bars - 1)
        sec = self.section(bar)
        prog = self.spec.progression[sec]
        return prog[(bar - SECTION_START[sec]) % len(prog)]

    def chord_tones(self, bar: int) -> tuple[int, int, int]:
        off, quality = DEGREE_INFO[self.degree(bar)]
        root = 60 + KEY_OFFSET[self.spec.key] + off
        while root > 67:
            root -= 12
        return (root, root + (4 if quality == "maj" else 3), root + 7)

    def bass_root(self, bar: int) -> int:
        off, _ = DEGREE_INFO[self.degree(bar)]
        return 36 + (KEY_OFFSET[self.spec.key] + off) % 12

    def melody_base(self) -> int:
        off = KEY_OFFSET[self.spec.key]
        return 72 + off if off <= 4 else 60 + off

    def hook_notes(self, bar: int) -> list[int]:
        if bar in CHORUS_BARS:
            return [self.melody_base() + SCALE[d - 1] for d in self.spec.hook[bar - CHORUS_BARS[0]]]
        if bar == 14:
            r, t3, t5 = self.chord_tones(14)
            return [t5, t3, r, t3]
        if bar == 15:
            return [self.melody_base()]
        return []


STEMS = ["kick", "snare", "hat", "clap", "crash", "bass", "inst", "melody", "mem0", "mem1", "mem2"]
PART_STEM = {"kick": "kick", "snare": "snare", "hat": "hat", "openHat": "hat", "clap": "clap"}


def _drums(song: Song, st: dict[str, np.ndarray]) -> None:
    sr = song.sr
    spec = song.spec
    sample = {"kick": synth.kick(sr), "snare": synth.snare(sr), "hat": synth.hat(sr),
              "openHat": synth.hat(sr, True), "clap": synth.clap(sr)}
    crash = synth.crash(sr)

    def hit(stem, smp, bar, beat, g=1.0):
        add_at(st[stem], smp, int(song.t(bar, beat) * sr), g)

    for bar in range(song.bars):
        pat = spec.drums[song.section(bar)]
        parts = ("hat",) if bar == 0 else DRUM_PARTS  # 첫 마디는 하이햇만으로 조용히 시작
        for part in parts:
            for i, ch in enumerate(pat[part]):
                if ch in VELOCITY:
                    g = VELOCITY[ch] * PART_GAIN[part] * (0.6 if bar == 0 else 1.0)
                    hit(PART_STEM[part], sample[part], bar, i / 4.0, g)
    fill_bars = {f["bar"] - 1: f["type"] for f in spec.fills}
    for bar, typ in fill_bars.items():
        if typ == "snare_roll":
            for i, b in enumerate((3, 3.25, 3.5, 3.75)):
                hit("snare", sample["snare"], bar, b, 0.45 + 0.15 * i)
            hit("crash", crash, bar + 1, 0, 0.7)
        elif typ == "tom_roll":
            for i, (b, f0) in enumerate(((3, 190.0), (3.25, 160.0), (3.5, 130.0), (3.75, 105.0))):
                hit("snare", synth.tom_tuned(sr, f0), bar, b, 0.7 + 0.08 * i)
            hit("crash", crash, bar + 1, 0, 0.7)
        elif typ == "crash":
            hit("crash", crash, bar, 0, 0.8)
        elif typ == "open_hat":
            hit("hat", sample["openHat"], bar, 2.5, 0.8)
            hit("hat", sample["openHat"], bar, 3.5, 0.6)
    for bar in getattr(song, "crash_bars", (2, 10)):  # 벌스, 후렴 시작
        if fill_bars.get(bar - 1) not in ("snare_roll", "tom_roll") and fill_bars.get(bar) != "crash":
            hit("crash", crash, bar, 0, 0.75)
    hit("kick", sample["kick"], song.bars, 0)
    hit("crash", crash, song.bars, 0, 0.9)


def _bass(song: Song, st: dict[str, np.ndarray]) -> None:
    sr = song.sr
    for bar in range(1, song.bars):
        root = song.bass_root(bar)
        sec = song.section(bar)
        if sec == "chorus":
            notes = [(0, 0.9), (1, 0.9), (2, 0.9), (3, 0.9)]
        elif bar == song.bars - 1:
            notes = [(0, 3.5)]
        else:
            notes = [(0, 1.5), (2, 1.5)]
        for beat, dur in notes:
            add_at(st["bass"], synth.bass_note(root, dur * song.spb, sr), int(song.t(bar, beat) * sr))
    add_at(st["bass"], synth.bass_note(36 + KEY_OFFSET[song.spec.key], 2.0, sr), int(song.t(song.bars, 0) * sr))


def _inst_pattern(style: str, tones: tuple[int, int, int]) -> list[tuple[float, list[int], float]]:
    r, t3, t5 = tones
    if style == "strum":  # 다운, 다운-업, 업-다운-업
        chord = [r, t3, t5, r + 12]
        return [(0, chord, 0.8), (1, chord, 0.75), (1.5, chord[::-1], 0.5), (2.5, chord[::-1], 0.5),
                (3, chord, 0.7), (3.5, chord[::-1], 0.5)]
    if style == "arp8":
        seq = [r - 12, t5 - 12, r, t3, t5, r, t3, t5 - 12]
        return [(i * 0.5, [m], 0.75 if i % 2 == 0 else 0.55) for i, m in enumerate(seq)]
    if style == "calypso":
        seq = [(0, r + 12), (0.75, t3 + 12), (1.5, t5), (2, t3 + 12), (2.75, r + 12), (3.5, t5)]
        return [(b, [m], 0.7) for b, m in seq]
    seq = [r + 12, t3 + 12, t5 + 12, r + 24]  # quarter
    return [(i, [m], 0.6) for i, m in enumerate(seq)]


def _instrument(song: Song, st: dict[str, np.ndarray]) -> None:
    sr = song.sr
    inst = song.spec.instrument_set
    style = song.spec.instrument_style
    for bar in range(song.bars):
        tones = song.chord_tones(bar)
        if bar == song.bars - 1:
            for i, m in enumerate(tones + (tones[0] + 12,)):
                add_at(st["inst"], synth.inst_note(inst, m, sr, i), int((song.t(bar, 0) + i * 0.015) * sr), 0.7)
        else:
            for beat, notes, vel in _inst_pattern(style, tones):
                base = song.t(bar, beat)
                for i, m in enumerate(notes):
                    add_at(st["inst"], synth.inst_note(inst, m, sr, bar + i), int((base + i * 0.012) * sr),
                           vel * (0.6 if bar == 0 else 1.0))
        for i, m in enumerate(song.hook_notes(bar)):
            add_at(st["melody"], synth.inst_note(inst, m, sr, i), int(song.t(bar, i) * sr), 0.8)
    r = 60 + KEY_OFFSET[song.spec.key]
    for i, m in enumerate((r, r + 4, r + 7, r + 12)):
        add_at(st["inst"], synth.inst_note(inst, m, sr, i), int((song.t(song.bars, 0) + i * 0.02) * sr), 0.8)


def _members(song: Song, st: dict[str, np.ndarray], members: list[tuple[str, str]]) -> None:
    sr = song.sr
    for idx, (mid, inst) in enumerate(members[:3]):
        smp = synth.MEMBER_SAMPLES[inst](sr)
        pats = song.spec.members.get(mid) or {}
        for bar in range(1, song.bars - 1):
            sec = "chorus" if song.section(bar) == "chorus" else "verse"
            pat = pats.get(sec, "x.......x.......")
            for i, ch in enumerate(pat):
                if ch in VELOCITY:
                    add_at(st[f"mem{idx}"], smp, int(song.t(bar, i / 4.0) * sr), VELOCITY[ch] * 0.85)


def render_backing(song: Song, members: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """members: [(memberId, 악기)]"""
    st = {name: np.zeros(song.n, np.float32) for name in STEMS}
    _drums(song, st)
    _bass(song, st)
    _instrument(song, st)
    _members(song, st, members)
    return st
