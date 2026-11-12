"""주행 데이터(속도 기록)를 비트 설정(BPM, 패턴, 필인 위치, 악기 세트)으로 바꾼다.

기획 문서의 매핑 초안을 그대로 따른다.
- 속도 0 / 1~30 / 31~70 / 71+ km/h -> BPM 80 / 95 / 110 / 125
- 가속 > +2 m/s^2 -> 드럼 필인, 감속 < -2 m/s^2 -> 하이햇 오픈
- 정차가 잦으면 셔플, 일정하게 달리면 4-on-the-floor
- 계절(날짜 + 바깥 온도) -> 악기 세트
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from .synth import SEASON_SETS

SONG_BARS = 16
EVENT_BARS = range(2, 14)  # 벌스와 후렴 마디에만 필인을 넣는다


@dataclass
class DriveProfile:
    scenario: str
    label: str
    bpm: int
    pattern: str  # "shuffle" 또는 "four_on_floor"
    season: str
    instrument_set: str
    events: dict[int, str] = field(default_factory=dict)  # 마디(0부터) -> "accel" / "decel"
    median_kmh: float = 0.0
    stops: int = 0


def _trace_city() -> np.ndarray:
    """시내: 출발, 짧게 달리고, 정차를 반복. 1초 간격 속도(km/h)."""
    v: list[float] = []
    cycles = [(3.5, 28, 9, 3.0, 6), (6.0, 25, 7, 4.0, 4), (3.0, 30, 10, 3.0, 7), (5.0, 22, 6, 5.0, 5), (3.5, 27, 8, 3.0, 4)]
    for acc_t, top, cruise, dec_t, stop in cycles:
        v += list(np.linspace(0, top, int(acc_t) + 1)[1:])
        v += [top + np.sin(i) for i in range(cruise)]
        v += list(np.linspace(top, 0, int(dec_t) + 1)[1:])
        v += [0.0] * stop
    return np.array(v)


def _trace_highway() -> np.ndarray:
    """고속도로: 90~100 km/h로 꾸준히, 추월 가속 한 번과 나들목 감속 한 번."""
    v = [95 + 3 * np.sin(i / 5) for i in range(40)]
    v += list(np.linspace(95, 117, 4)[1:])
    v += [115 + 2 * np.sin(i / 3) for i in range(30)]
    v += [112 - i for i in range(10)]
    v += list(np.linspace(102, 72, 5)[1:])
    v += [75 + 2 * np.sin(i / 4) for i in range(25)]
    return np.array(v)


def _trace_suburb() -> np.ndarray:
    v = [45 + 8 * np.sin(i / 6) for i in range(60)]
    v += list(np.linspace(50, 30, 4)[1:])
    v += [30 + 3 * np.sin(i / 3) for i in range(20)]
    v += list(np.linspace(30, 58, 4)[1:])
    v += [58 + 2 * np.sin(i / 5) for i in range(30)]
    return np.array(v)


SCENARIOS = {
    "city": ("시내 주행 (정차 잦음)", _trace_city),
    "highway": ("고속도로 (일정한 고속)", _trace_highway),
    "suburb": ("교외 도로 (중간 속도)", _trace_suburb),
}


def speed_to_bpm(kmh: float) -> int:
    if kmh < 1:
        return 80
    if kmh <= 30:
        return 95
    if kmh <= 70:
        return 110
    return 125


def season_of(date: _dt.date | None = None, outside_c: float | None = None) -> str:
    date = date or _dt.date.today()
    m = date.month
    s = "spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8) else "autumn" if m in (9, 10, 11) else "winter"
    if outside_c is not None:
        if outside_c >= 25:
            s = "summer"
        elif outside_c <= 3:
            s = "winter"
    return s


def analyze_trace(speeds_kmh: np.ndarray, dt: float = 1.0, bars: int = SONG_BARS,
                  event_bars: range = EVENT_BARS) -> tuple[int, str, dict[int, str], float, int]:
    v = np.asarray(speeds_kmh, dtype=float)
    moving = v[v >= 1]
    median = float(np.median(moving)) if moving.size else 0.0
    bpm = speed_to_bpm(median)
    stopped = v < 1
    stops = int(np.sum(stopped[1:] & ~stopped[:-1]))
    minutes = max(len(v) * dt / 60.0, 1e-6)
    pattern = "shuffle" if stops / minutes >= 1.5 else "four_on_floor"
    acc = np.diff(v / 3.6) / dt
    events: dict[int, str] = {}
    last_bar = -9
    for i, a in enumerate(acc):
        kind = "accel" if a > 2.0 else "decel" if a < -2.0 else None
        if kind is None:
            continue
        bar = int((i + 1) / len(v) * bars)
        bar = min(max(bar, event_bars.start), event_bars.stop - 1)
        if bar in events or bar - last_bar < 2:
            continue
        events[bar] = kind
        last_bar = bar
    return bpm, pattern, events, median, stops


def from_scenario(name: str, season: str | None = None, outside_c: float | None = None) -> DriveProfile:
    if name not in SCENARIOS:
        raise ValueError(f"모르는 주행 상황: {name} (가능: {', '.join(SCENARIOS)})")
    label, maker = SCENARIOS[name]
    bpm, pattern, events, median, stops = analyze_trace(maker())
    s = season or season_of(outside_c=outside_c)
    return DriveProfile(name, label, bpm, pattern, s, SEASON_SETS[s], events, median, stops)


def from_speeds(speeds_kmh, dt: float = 1.0, season: str | None = None, outside_c: float | None = None,
                bars: int = SONG_BARS, event_bars: range = EVENT_BARS) -> DriveProfile:
    """앱이 보내 준 최근 속도 기록으로 만든다 (서버 연동용)."""
    bpm, pattern, events, median, stops = analyze_trace(np.asarray(speeds_kmh, dtype=float), dt, bars, event_bars)
    s = season or season_of(outside_c=outside_c)
    return DriveProfile("live", "실제 주행 기록", bpm, pattern, s, SEASON_SETS[s], events, median, stops)
