"""서버용 곡 만들기.

녹음 한 번(가족 여러 명)에서 고른 단어와 의성어로 28마디 노래를 만들고,
앱이 가사를 따라 강조할 수 있게 줄마다 시작·끝 시간과 강조 위치를 함께 돌려준다.

곡 구조: 인트로 2 -> 벌스 8 -> 후렴 4 -> 벌스 8 -> 후렴 4 -> 아웃트로 2 (28마디)
가사 한 줄 = 2마디. 벌스 줄은 "사과는 빨갛다, 둥둥 둥둥"처럼
앞 마디에서 TTS가 "사과는 빨갛다"를 부르고, 뒤 마디에서 아이 목소리가 "둥둥 둥둥"을 받는다.
"""
from __future__ import annotations

import datetime as dt
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal

from . import synth
from .arrange import Song, _bass, _drums, _instrument
from .beat import DEFAULT_HOOK, DEFAULT_PROG, DEFAULT_STYLE, SCALE, BeatSpec, _pat
from .drive import DriveProfile, from_speeds, season_of
from .dsp import SR, add_at, compress, db_to_amp, envelope, filt, limit, pan, reverb_ir
from .lyrics import josa
from .mix import active_normalize
from .text import Analysis, Noun, Ono, Word, fit_title, pick_nouns, pick_onos
from .tts import Voice
from .vocals import Clip, Placer, _targets, extract_clip

FORM = [("intro", 2), ("verse", 8), ("chorus", 4), ("verse", 8), ("chorus", 4), ("outro", 2)]
EVENT_BARS = range(2, 26)  # 필인은 벌스·후렴 마디에만

# 앱의 Instrument 이름 -> 소리
PERCUSSION = {"DRUM": synth.tom, "TAMBOURINE": synth.tambourine, "SHAKER": synth.shaker,
              "WOODBLOCK": synth.woodblock, "MARACAS": synth.maracas}
PITCHED_REGISTER = {"BELL": 84, "UKULELE": 67, "STEELDRUM": 76, "GUITAR": 60, "XYLOPHONE": 79, "HARP": 72, "PIANO": 60}
INSTRUMENT_LABELS = {"DRUM": "북", "TAMBOURINE": "탬버린", "SHAKER": "셰이커", "BELL": "벨", "UKULELE": "우쿨렐레",
                     "STEELDRUM": "스틸드럼", "GUITAR": "기타", "XYLOPHONE": "실로폰", "WOODBLOCK": "우드블록",
                     "HARP": "하프", "PIANO": "피아노", "MARACAS": "마라카스"}
EIGHTHS = list(range(0, 16, 2))
MEMBER_PATTERNS = {
    "DRUM": {"verse": _pat([0, 8], [0]), "chorus": _pat([0, 4, 8, 12], [0, 8])},
    "TAMBOURINE": {"verse": _pat([4, 12]), "chorus": _pat([2, 4, 6, 10, 12, 14], [4, 12])},
    "SHAKER": {"verse": _pat(EIGHTHS), "chorus": _pat(EIGHTHS + [7, 15], [2, 6, 10, 14])},
    "BELL": {"verse": _pat([0, 8]), "chorus": _pat([0, 4, 8, 12], [0])},
    "UKULELE": {"verse": _pat([2, 6, 10, 14]), "chorus": _pat([2, 6, 10, 14], [6, 14])},
    "STEELDRUM": {"verse": _pat([0, 3, 6, 10]), "chorus": _pat([0, 3, 6, 8, 10, 14])},
    "GUITAR": {"verse": _pat([0, 4, 8, 12]), "chorus": _pat(EIGHTHS)},
    "XYLOPHONE": {"verse": _pat([0, 6, 8, 14]), "chorus": _pat(EIGHTHS)},
    "WOODBLOCK": {"verse": _pat([0, 6, 8, 12]), "chorus": _pat([0, 3, 6, 8, 12, 14], [0, 8])},
    "MARACAS": {"verse": _pat(EIGHTHS), "chorus": _pat(EIGHTHS + [3, 11], [0, 8])},
    "HARP": {"verse": _pat([0, 4, 8, 12]), "chorus": _pat(EIGHTHS, [0, 8])},
    "PIANO": {"verse": _pat([0, 8]), "chorus": _pat([0, 4, 8, 12], [0, 8])},
}
MEMBER_LEVEL_DB = {"DRUM": -13.0, "TAMBOURINE": -9.0, "SHAKER": -12.0, "BELL": -14.0, "UKULELE": -12.0,
                   "STEELDRUM": -13.0, "GUITAR": -12.0, "XYLOPHONE": -13.0, "WOODBLOCK": -13.0, "MARACAS": -12.0,
                   "HARP": -13.0, "PIANO": -12.0}
MEMBER_PANS = [-0.5, 0.5, -0.25, 0.25]
BASE_LEVELS = {
    "kick": (-10.0, 0.0), "snare": (-3.0, 0.05), "hat": (-8.0, 0.3), "clap": (-3.0, -0.1),
    "crash": (-12.0, 0.2), "bass": (-16.0, 0.0), "inst": (-5.0, -0.3), "melody": (-5.0, 0.2),
}
DUCKED = {"inst", "melody", "bass"}

# 아이들이 자주 말하는 단어는 그 단어에 어울리는 서술 두 개, 나머지는 종류별 기본 서술
PREDICATES = {
    "사과": ("빨갛다", "맛있다"), "딸기": ("빨갛다", "달콤해"), "바나나": ("노랗다", "길쭉해"), "수박": ("커다랗다", "시원해"),
    "포도": ("동글동글", "달콤해"), "참외": ("노랗다", "달콤해"), "복숭아": ("말랑해", "달콤해"), "과자": ("바삭해", "맛있다"),
    "사탕": ("달콤해", "알록달록"), "우유": ("하얗다", "고소해"), "케이크": ("달콤해", "맛있다"), "아이스크림": ("시원해", "달콤해"),
    "하늘": ("파랗다", "높아요"), "바다": ("파랗다", "넓어요"), "구름": ("하얗다", "폭신해"), "무지개": ("알록달록", "예뻐요"),
    "나무": ("커다랗다", "초록색"), "자동차": ("빠르다", "부릉부릉"), "기차": ("길어요", "칙칙폭폭"), "비행기": ("높이 날아", "슝슝 날아"),
    "버스": ("커다랗다", "부릉부릉"), "자전거": ("씽씽 달려", "따르릉"), "풍선": ("둥실둥실", "동글동글"), "로봇": ("멋있다", "삐삐삐"),
    "강아지": ("귀여워", "멍멍해"), "고양이": ("귀여워", "야옹해"), "토끼": ("깡충깡충", "귀가 길어"), "공룡": ("커다랗다", "쿵쿵 걸어"),
    "사자": ("멋있다", "어흥해"), "오리": ("뒤뚱뒤뚱", "꽥꽥해"), "병아리": ("노랗다", "삐약삐약"), "물고기": ("헤엄쳐요", "뻐끔뻐끔"),
    "엄마": ("최고야", "사랑해"), "아빠": ("최고야", "힘이 세"), "할머니": ("다정해", "좋아요"), "할아버지": ("멋있다", "좋아요"),
    "친구": ("좋아요", "반가워"), "아기": ("귀여워", "방긋 웃어"), "언니": ("좋아요", "멋있다"), "오빠": ("좋아요", "멋있다"),
    "공원": ("넓어요", "신나요"), "놀이터": ("신나요", "재밌어"), "유치원": ("즐거워", "재밌어"), "학교": ("즐거워", "재밌어"),
}
GENERIC_PREDICATES = {
    "object": [("반짝반짝", "좋아요"), ("최고야", "신나요")],
    "animal": [("귀여워", "좋아요"), ("깡충깡충", "신나요")],
    "person": [("최고야", "사랑해"), ("좋아요", "고마워")],
    "place": [("신나요", "가고 싶어"), ("좋아요", "재밌어")],
}
DRIVE_LINES = ["부릉부릉 출발해요", "창밖을 보아요", "바람이 불어와요", "신나게 달려요"]
DEFAULT_MAIN, DEFAULT_MELODY = "둥둥", "랄랄라"


@dataclass
class MemberData:
    member_id: int
    instrument: str
    words: list[Word]
    analysis: Analysis
    speech_sec: float = 0.0


@dataclass
class Line:
    kind: str  # verse / chorus1 / chorus2 / outro
    bar: int  # 시작 마디 (0부터)
    section: str
    text: str
    sung: str = ""  # TTS가 부를 앞부분
    slot: str = "main"  # 뒷부분 의성어: main / melody
    noun: Noun | None = None
    extra_nouns: list[Noun] = field(default_factory=list)


class FormSong(Song):
    """섹션이 되풀이되는 곡 구조. 코드·드럼·후렴 멜로디는 섹션 안에서의 마디 번호로 고른다."""

    def __init__(self, spec: BeatSpec, form=FORM, sr: int = SR, tail: float = 2.5):
        bars = sum(n for _, n in form)
        super().__init__(spec, bars=bars, sr=sr, tail=tail)
        self.form = form
        self.bar_info: list[tuple[str, int]] = []
        self.section_starts: list[tuple[str, int, int]] = []
        b = 0
        for name, n in form:
            self.section_starts.append((name, b, n))
            self.bar_info += [(name, i) for i in range(n)]
            b += n
        self.crash_bars = tuple(start for name, start, _ in self.section_starts if name in ("verse", "chorus"))

    def section(self, bar: int) -> str:
        if bar < 0:
            return self.bar_info[0][0]
        return self.bar_info[bar][0] if bar < self.bars else "outro"

    def degree(self, bar: int) -> str:
        name, idx = self.bar_info[min(max(bar, 0), self.bars - 1)]
        prog = self.spec.progression[name]
        return prog[idx % len(prog)]

    def hook_notes(self, bar: int) -> list[int]:
        if not 0 <= bar < self.bars:
            return []
        name, idx = self.bar_info[bar]
        if name == "chorus":
            return [self.melody_base() + SCALE[d - 1] for d in self.spec.hook[idx % len(self.spec.hook)]]
        if name == "outro" and idx == 0:
            r, t3, t5 = self.chord_tones(bar)
            return [t5, t3, r, t3]
        if name == "outro":
            return [self.melody_base()]
        return []

    @property
    def music_end(self) -> float:
        return self.bars * 4 * self.spb


# ---------- 비트 악보 ----------
def beat_type_for_bpm(bpm: int) -> int:
    return 2 if bpm <= 80 else 4 if bpm <= 95 else 8 if bpm <= 110 else 16


def drums_for(beat_type: int) -> dict[str, dict[str, str]]:
    q = [0, 4, 8, 12]
    if beat_type == 2:
        verse = {"kick": _pat([0], [0]), "snare": _pat([8]), "hat": _pat(q), "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, kick=_pat([0, 10], [0]), clap=_pat([8]))
    elif beat_type == 4:
        verse = {"kick": _pat([0, 8], [0]), "snare": _pat([4, 12]), "hat": _pat(q), "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, hat=_pat(EIGHTHS, q), clap=_pat([4, 12]))
    elif beat_type == 8:
        verse = {"kick": _pat([0, 8, 10], [0, 8]), "snare": _pat([4, 12]), "hat": _pat(EIGHTHS, q),
                 "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, openHat=_pat([6, 14]), clap=_pat([4, 12]))
    else:
        verse = {"kick": _pat(q, [0]), "snare": _pat([4, 12]), "hat": _pat(range(16), EIGHTHS),
                 "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, hat=_pat(q), openHat=_pat([2, 6, 10, 14]), clap=_pat([4, 12]))
    outro = {"kick": _pat([0]), "snare": _pat([8]), "hat": _pat([0, 8]), "openHat": _pat([]), "clap": _pat([])}
    return {"intro": dict(verse), "verse": verse, "chorus": chorus, "outro": outro}


def _parse_date(text: str | None) -> dt.date | None:
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def make_spec(beat: dict | None, drive: dict | None, bars: int, today: dt.date | None = None
              ) -> tuple[BeatSpec, DriveProfile | None, int, dt.date]:
    """앱이 S03에서 보여 준 BPM·비트를 그대로 쓰고, 속도 기록으로 필인과 셔플을, 날짜·온도로 악기를 정한다."""
    beat = beat or {}
    drive = drive or {}
    try:
        bpm = int(round(float(beat.get("bpm") or 0)))
    except (TypeError, ValueError):
        bpm = 0
    speeds = [float(v) for v in (drive.get("speedKmh") or []) if isinstance(v, (int, float))]
    outside = drive.get("outsideTempC")
    outside = float(outside) if isinstance(outside, (int, float)) else None
    day = _parse_date(drive.get("startedAt")) or today or dt.date.today()
    season = season_of(day, outside)
    prof = None
    if speeds:
        prof = from_speeds(speeds, float(drive.get("intervalSec") or 1.0), season, outside, bars=bars, event_bars=EVENT_BARS)
        if not bpm:
            bpm = prof.bpm
    bpm = min(140, max(70, bpm or 110))
    try:
        beat_type = int(beat.get("beatType") or beat.get("beatsPerBar") or 0)
    except (TypeError, ValueError):
        beat_type = 0
    if beat_type not in (2, 4, 8, 16):
        beat_type = beat_type_for_bpm(bpm)
    shuffle = bool(prof and prof.pattern == "shuffle" and beat_type in (4, 8))
    inst_set = synth.SEASON_SETS[season]
    fills = []
    if prof:
        fills = [{"bar": b + 1, "type": "snare_roll" if e == "accel" else "open_hat",
                  "why": "가속 구간" if e == "accel" else "감속 구간"} for b, e in sorted(prof.events.items())][:6]
    spec = BeatSpec(
        bpm=bpm, swing=1.0 if shuffle else 0.0, key="C", instrument_set=inst_set,
        instrument_style=DEFAULT_STYLE[inst_set], progression={k: list(v) for k, v in DEFAULT_PROG.items()},
        drums=drums_for(beat_type), fills=fills, members={}, hook=[list(h) for h in DEFAULT_HOOK],
        mood="신나는 셔플" if shuffle else f"{beat_type}비트",
        reason=f"S03에서 보여 준 {bpm} BPM·{beat_type}비트, {season} 악기" + (", 정차가 잦아 셔플" if shuffle else ""),
        source="rule",
    )
    return spec, prof, beat_type, day


# ---------- 반주 ----------
def playable(inst: str) -> str:
    """모르는 악기 이름은 북 소리로 대신한다 (앱에 악기가 늘어나도 서버가 멈추지 않게)."""
    return inst if inst in PERCUSSION or inst in PITCHED_REGISTER else "DRUM"


def member_note(inst: str, midi: int, sr: int) -> np.ndarray:
    if inst == "BELL":
        return synth.bell(midi, 1.0, sr)
    if inst == "UKULELE":
        return synth.pluck(midi, 0.5, sr, 0.99, 0.7, 1)
    if inst == "STEELDRUM":
        return synth.steel(midi, 0.6, sr)
    if inst == "GUITAR":
        return synth.pluck(midi, 0.9, sr, 0.995, 0.5, 2)
    if inst == "HARP":
        return synth.pluck(midi, 1.4, sr, 0.997, 0.85, 3)
    if inst == "PIANO":
        return synth.piano(midi, 1.2, sr)
    return synth.xylo(midi, 0.5, sr)


def member_hit(inst: str, sr: int, midi: int | None = None) -> np.ndarray:
    """악기 한 번 치는 소리 (앱의 S04 소리·미리 듣기용으로도 쓴다)."""
    if inst in PERCUSSION:
        return PERCUSSION[inst](sr)
    return member_note(inst, midi or PITCHED_REGISTER.get(inst, 72), sr)


def _register(tone: int, base: int) -> int:
    while tone < base:
        tone += 12
    while tone >= base + 12:
        tone -= 12
    return tone


def render_backing(song: FormSong, members: list[tuple[int, str]]) -> dict[str, np.ndarray]:
    """members: [(memberId, 앱 악기 이름)] 최대 4명."""
    names = ["kick", "snare", "hat", "clap", "crash", "bass", "inst", "melody"] + [f"m{i}" for i in range(4)]
    st = {n: np.zeros(song.n, np.float32) for n in names}
    _drums(song, st)
    _bass(song, st)
    _instrument(song, st)
    vel = {"x": 0.75, "X": 1.0}
    for idx, (_mid, inst) in enumerate(members[:4]):
        pats = MEMBER_PATTERNS.get(inst, MEMBER_PATTERNS["DRUM"])
        arp = 0
        for bar in range(1, song.bars - 1):
            sec = "chorus" if song.section(bar) == "chorus" else "verse"
            if song.section(bar) == "intro" and bar == 0:
                continue
            tones = song.chord_tones(bar)
            for i, ch in enumerate(pats[sec]):
                if ch not in vel:
                    continue
                if inst in PERCUSSION:
                    smp = PERCUSSION[inst](song.sr)
                else:
                    smp = member_note(inst, _register(tones[arp % 3], PITCHED_REGISTER[inst]), song.sr)
                    arp += 1
                add_at(st[f"m{idx}"], smp, int(song.t(bar, i / 4.0) * song.sr), vel[ch] * 0.85)
    return st


def mix_song(stems: dict[str, np.ndarray], member_insts: list[str], vocal: np.ndarray, sr: int) -> np.ndarray:
    n = len(vocal)
    out = np.zeros((n, 2), np.float32)
    v = compress(vocal, sr, -22.0, 3.0, 90.0)
    v = active_normalize(v, sr, -16.0)
    venv = envelope(v, sr, 60.0)
    venv = venv / (venv.max() + 1e-9)
    duck = 1.0 - 0.45 * np.clip(venv * 1.6, 0.0, 1.0)
    levels = dict(BASE_LEVELS)
    for i, inst in enumerate(member_insts[:4]):
        levels[f"m{i}"] = (MEMBER_LEVEL_DB.get(inst, -12.0), MEMBER_PANS[i])
    for name, x in stems.items():
        if name not in levels or not np.any(x):
            continue
        db, p = levels[name]
        y = x[:n] * db_to_amp(db)
        if name in DUCKED:
            y = y * duck[: len(y)]
        out[: len(y)] += pan(y, p)
    out += pan(v, 0.0)
    irl, irr = reverb_ir(sr, 1.1, 0.9, 11), reverb_ir(sr, 1.1, 0.9, 12)
    send = filt(v, sr, "highpass", 200) * db_to_amp(-15.0)
    wet = np.stack([signal.fftconvolve(send, irl)[:n], signal.fftconvolve(send, irr)[:n]], axis=1)
    out += wet.astype(np.float32)
    out = np.stack([filt(out[:, 0], sr, "highpass", 30), filt(out[:, 1], sr, "highpass", 30)], axis=1)
    level = np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-12)
    out = out * (db_to_amp(-14.5) / level)
    return limit(out, sr, -1.5, 3.0, 80.0)


# ---------- 가사 ----------
def echo_of(ono: str) -> str:
    """'둥둥' -> '둥둥', '랄랄라' -> '랄라', '부릉부릉' -> '부릉'."""
    n = len(ono)
    return ono if n <= 2 else ono[-2:] if n == 3 else ono[n // 2:]


def predicates(noun: Noun) -> tuple[str, str]:
    if noun.text in PREDICATES:
        return PREDICATES[noun.text]
    cands = GENERIC_PREDICATES[noun.category]
    return cands[zlib.crc32(noun.text.encode("utf-8")) % len(cands)]


def build_lines(song: FormSong, nouns: list[Noun], main: str, melody: str, title: str) -> list[Line]:
    verse_starts = [start for name, start, _ in song.section_starts if name == "verse"]
    chorus_starts = [start for name, start, _ in song.section_starts if name == "chorus"]
    outro_start = next(start for name, start, _ in song.section_starts if name == "outro")
    lines: list[Line] = []

    def verse_line(bar: int, sung: str, slot: str, noun: Noun | None, extra: list[Noun] | None = None) -> Line:
        tail = f"{main} {main}" if slot == "main" else f"{melody} {echo_of(melody)}"
        return Line("verse", bar, "verse", f"{sung}, {tail}", sung, slot, noun, extra or [])

    plans: list[list[tuple[str, Noun | None, list[Noun]]]] = []
    if nouns:
        first, second = nouns[0], (nouns[1] if len(nouns) > 1 else nouns[0])
        third, fourth = (nouns[2] if len(nouns) > 2 else None), (nouns[3] if len(nouns) > 3 else None)
        a1, a2 = predicates(first)
        b1, b2 = predicates(second)
        verse_a = [(f"{josa(first.text, '은/는')} {a1}", first, []), (f"{josa(first.text, '은/는')} {a2}", first, []),
                   (f"{josa(second.text, '은/는')} {b1}", second, []), (f"{josa(second.text, '은/는')} {b2}", second, [])]
        if third is not None:
            c1, c2 = predicates(third)
            d = fourth or third
            d1, d2 = predicates(d)
            verse_b = [(f"{josa(third.text, '은/는')} {c1}", third, []), (f"{josa(third.text, '은/는')} {c2}", third, []),
                       (f"{josa(d.text, '은/는')} {d1}", d, []), (f"{josa(d.text, '은/는')} {d2}", d, [])]
        else:
            verse_b = [(f"{josa(first.text, '이랑/랑')} 같이 노래해", first, []), (f"{second.text}도 신이 나요", second, []),
                       (f"{josa(first.text, '은/는')} {a1}", first, []), (f"{josa(second.text, '은/는')} {b2}", second, [])]
        plans = [verse_a, verse_b]
    else:
        plans = [[(t, None, []) for t in DRIVE_LINES], [(t, None, []) for t in DRIVE_LINES]]

    for k, start in enumerate(verse_starts):
        plan = plans[k % len(plans)]
        for j, (sung, noun, extra) in enumerate(plan[:4]):
            lines.append(verse_line(start + 2 * j, sung, "main" if j % 2 == 0 else "melody", noun, extra))
        if k < len(chorus_starts):
            c = chorus_starts[k]
            lines.append(Line("chorus1", c, "chorus", f"{main} {main}, {title}", title, "main"))
            lines.append(Line("chorus2", c + 2, "chorus", f"{melody} 신나는 우리 노래", "신나는 우리 노래", "melody"))
    lines.append(Line("outro", outro_start, "outro", f"{title}, {melody} {echo_of(melody)}", title, "melody"))
    lines.sort(key=lambda ln: ln.bar)
    return lines


def make_title(nouns: list[Noun], day: dt.date) -> str:
    return fit_title(f"{nouns[0].text} 노래") if nouns else f"{day.month}월 {day.day}일 노래"


# ---------- 아이 목소리 조각 ----------
def _neighbors(words: list[Word], start: float, end: float) -> tuple[float | None, float | None]:
    prev_end, next_start = None, None
    for w in words:
        if w.end <= start + 1e-3:
            prev_end = w.end if prev_end is None else max(prev_end, w.end)
        elif w.start >= end - 1e-3:
            next_start = w.start if next_start is None else min(next_start, w.start)
    return prev_end, next_start


def best_clip(audio: np.ndarray, members: list[MemberData], text: str, prefer: str | None, kind: str) -> Clip | None:
    all_words = sorted((w for m in members for w in m.words), key=lambda w: w.start)
    cands = []
    for m in members:
        items = m.analysis.nouns if kind == "noun" else m.analysis.onos
        for it in items:
            if it.text == text:
                whole = (it.frac_end - it.frac_start) > 0.98
                cands.append((0 if it.member == prefer else 1, 0 if whole else 1, it.start, it))
    cands.sort(key=lambda c: c[:3])
    for _, _, _, it in cands:
        pe, ns = _neighbors(all_words, it.start, it.end)
        c = extract_clip(audio, SR, it.start, it.end, it.frac_start, it.frac_end, pe, ns, it.text, it.member)
        if c is not None:
            return c
    return None


# ---------- 보컬 ----------
def render_vocals(song: FormSong, lines: list[Line], noun_clips: dict[str, Clip | None],
                  ono_clips: dict[str, Clip | None], main: str, melody: str, voice: Voice | None
                  ) -> tuple[np.ndarray, list[dict]]:
    p = Placer(song)
    main_c, mel_c, extra_c = ono_clips.get("main"), ono_clips.get("melody"), ono_clips.get("extra")

    def child_or_tts(clip: Clip | None, text: str, bar: int, beat: float, target: int | None, gain: float = 0.9) -> None:
        if not p.clip(clip, bar, beat, target, gain) and voice is not None:
            p.say(voice, text, bar, beat, 1.0, gain)

    child_or_tts(main_c, main, 1, 0, _targets(song, 1, 0))
    child_or_tts(main_c, main, 1, 2, _targets(song, 1, 2))
    child_or_tts(mel_c or extra_c, melody, 1, 3, _targets(song, 1, 1))
    for ln in lines:
        b = ln.bar
        if ln.kind == "verse":
            end = p.chant(voice, ln.sung, b, 0, 7) if voice is not None else 0.0
            nc = noun_clips.get(ln.noun.text) if ln.noun is not None else None
            if nc is not None and end <= 3.0:
                p.clip(nc, b, 3.0, _targets(song, b, 0), 0.8)
            if ln.slot == "main":
                child_or_tts(main_c, main, b + 1, 0, _targets(song, b + 1, 0))
                child_or_tts(main_c, main, b + 1, 1, _targets(song, b + 1, 2))
            else:
                child_or_tts(mel_c or main_c, melody, b + 1, 0, _targets(song, b + 1, 0))
                child_or_tts(mel_c or main_c, melody, b + 1, 2, _targets(song, b + 1, 1), 0.75)
        elif ln.kind == "chorus1":
            n1, n2 = song.hook_notes(b), song.hook_notes(b + 1)
            child_or_tts(main_c, main, b, 0, n1[0])
            child_or_tts(main_c, main, b, 1, n1[1])
            if voice is not None:
                p.chant(voice, ln.sung, b, 2, 4)
            child_or_tts(main_c, main, b + 1, 0, n2[0])
            child_or_tts(extra_c or main_c, main, b + 1, 2, n2[2], 0.8)
        elif ln.kind == "chorus2":
            n1, n2 = song.hook_notes(b), song.hook_notes(b + 1)
            child_or_tts(mel_c or main_c, melody, b, 0, n1[0])
            if voice is not None:
                p.chant(voice, ln.sung, b, 1.5, 5)
            child_or_tts(mel_c or main_c, melody, b + 1, 0, n2[0])
            child_or_tts(main_c, main, b + 1, 2, n2[2], 0.8)
        elif ln.kind == "outro":
            if voice is not None:
                p.chant(voice, ln.sung, b, 0, 4)
            child_or_tts(mel_c or main_c, melody, b, 3, song.hook_notes(b)[3])
            child_or_tts(main_c, main, b + 1, 0, song.melody_base())
    p.clip(main_c, song.bars, 0, song.melody_base(), 0.8)
    return p.buf, p.log


# ---------- 결과 ----------
def _member_int(label: str | None) -> int | None:
    try:
        return int(label) if label not in (None, "") else None
    except ValueError:
        return None


def highlights_for(ln: Line, slot_onos: dict[str, Ono | None]) -> list[dict]:
    out: list[dict] = []
    taken: list[tuple[int, int]] = []

    def add(start: int, end: int, member: int | None, kind: str) -> None:
        if member is None or any(not (end <= a or start >= b) for a, b in taken):
            return
        taken.append((start, end))
        out.append({"start": start, "end": end, "memberId": member, "kind": kind})

    for n in [ln.noun] + ln.extra_nouns:
        if n is None:
            continue
        idx = ln.text.find(n.text)
        if idx >= 0:
            add(idx, idx + len(n.text), _member_int(n.member), "word")
    for o in slot_onos.values():
        if o is None:
            continue
        start = 0
        while True:
            idx = ln.text.find(o.text, start)
            if idx < 0:
                break
            add(idx, idx + len(o.text), _member_int(o.member), "onomatopoeia")
            start = idx + len(o.text)
    return sorted(out, key=lambda h: h["start"])


def make_session_song(audio44: np.ndarray, members: list[MemberData], all_members: list[tuple[int, str]],
                      drive: dict | None, beat: dict | None, voice: Voice | None,
                      progress: Callable[[str, float], None] = lambda s, p: None,
                      log: Callable[[str], None] = print, today: dt.date | None = None
                      ) -> tuple[np.ndarray, dict]:
    """녹음 한 번 -> (스테레오 음원, 결과 JSON). all_members는 요청에 온 가족 전체 [(memberId, 악기)]."""
    progress("WRITE_LYRICS", 0.35)
    analyses = [m.analysis for m in members]
    nouns = pick_nouns(analyses)
    onos = pick_onos(analyses)
    main = onos["main"].text if onos.get("main") else DEFAULT_MAIN
    melody = onos["melody"].text if onos.get("melody") else DEFAULT_MELODY
    if melody == main:
        melody = DEFAULT_MELODY if main != DEFAULT_MELODY else DEFAULT_MAIN

    spec, prof, beat_type, day = make_spec(beat, drive, sum(n for _, n in FORM), today)
    title = make_title(nouns, day)
    song = FormSong(spec)
    lines = build_lines(song, nouns, main, melody, title)
    log(f"가사 단어: {', '.join(n.text for n in nouns) or '없음'} / 의성어: {main}, {melody} / 제목: {title}")

    progress("SPLIT_VOICES", 0.5)
    noun_clips = {n.text: best_clip(audio44, members, n.text, n.member, "noun") for n in nouns}
    ono_clips = {k: best_clip(audio44, members, o.text, o.member, "ono") for k, o in onos.items()}
    missing = [t for t, c in noun_clips.items() if c is None] + [o.text for k, o in onos.items() if ono_clips.get(k) is None]
    if missing:
        log(f"  잘라 내지 못한 소리: {', '.join(missing)}")

    progress("ADD_INSTRUMENTS", 0.65)
    known = {mid for mid, _ in all_members}
    spoke = [(m.member_id, playable(m.instrument)) for m in members if m.member_id in known and m.speech_sec >= 0.5]
    players = spoke or [(mid, playable(inst)) for mid, inst in all_members]
    stems = render_backing(song, players)
    vox, placed = render_vocals(song, lines, noun_clips, ono_clips, main, melody, voice)
    stereo = mix_song(stems, [inst for _, inst in players], vox, song.sr)

    progress("FINISH", 0.9)
    slot_onos = {"main": onos.get("main"), "melody": onos.get("melody")}
    end_ms = int(round(song.music_end * 1000))
    lyric_items = []
    for i, ln in enumerate(lines):
        start = int(round(song.t(ln.bar, 0) * 1000))
        nxt = int(round(song.t(lines[i + 1].bar, 0) * 1000)) if i + 1 < len(lines) else end_ms
        lyric_items.append({"text": ln.text, "startMs": start, "endMs": nxt, "section": ln.section,
                            "highlights": highlights_for(ln, slot_onos)})
    heard_nouns, seen = [], set()
    for a in analyses:
        for n in a.nouns:
            if n.text not in seen:
                seen.add(n.text)
                heard_nouns.append({"text": n.text, "memberId": _member_int(n.member)})
    heard_onos, seen = [], set()
    for a in analyses:
        for o in a.onos:
            if o.text not in seen:
                seen.add(o.text)
                heard_onos.append({"text": o.text, "memberId": _member_int(o.member)})
    speech = {m.member_id: m.speech_sec for m in members}
    result = {
        "title": title,
        "durationMs": int(round(len(stereo) / song.sr * 1000)),
        "bpm": spec.bpm,
        "beatType": beat_type,
        "firstBeatMs": 0,
        "lyrics": lyric_items,
        "words": heard_nouns[:8],
        "onomatopoeia": heard_onos[:8],
        "members": [{"memberId": mid, "instrument": inst, "speechSec": round(float(speech.get(mid, 0.0)), 1)}
                    for mid, inst in all_members],
        "engine": {"form": [f"{n}:{b}" for n, b in FORM], "instrumentSet": spec.instrument_set, "swing": spec.swing,
                   "fills": spec.fills, "season": season_of(day), "vocalEvents": len(placed),
                   "drive": ({"medianKmh": round(prof.median_kmh, 1), "stops": prof.stops, "pattern": prof.pattern}
                             if prof else None)},
    }
    return stereo, result
