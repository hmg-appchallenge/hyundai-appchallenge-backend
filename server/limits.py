"""노래 만들기 횟수 제한.

앱에 넣은 팀 키는 APK를 풀면 꺼낼 수 있어서 숨길 수 없다. 키가 새어 나가도 바깥 AI(Claude·ElevenLabs)
요금이 끝없이 나가지 않게, 서버가 하루에 만드는 노래 수와 한 곳(IP)에서 한 시간에 만드는 노래 수를 묶는다.

- KR_SONGS_PER_DAY: 하루 전체 (기본 60곡, 0이면 제한 없음). 날짜는 한국 시간 자정에 바뀐다
- KR_SONGS_PER_HOUR_PER_IP: 같은 IP에서 한 시간에 (기본 10곡, 0이면 제한 없음)

IP는 프록시 머리글에서 읽어서 속일 수 있다. 그래서 IP 제한은 보조이고, 돈을 지키는 건 하루 전체 제한이다.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
DEFAULT_PER_DAY = 60
DEFAULT_PER_HOUR_PER_IP = 10


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def client_ip(headers, fallback: str | None) -> str:
    """프록시(Cloudflare, ngrok, 허깅페이스)를 거쳐 온 요청이면 원래 보낸 곳의 IP를 고른다."""
    for name in ("cf-connecting-ip", "x-real-ip"):
        value = headers.get(name)
        if value:
            return value.strip()
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"


class SongLimiter:
    def __init__(self, per_day: int | None = None, per_hour_per_ip: int | None = None, clock=time.time):
        self.per_day = _env_int("KR_SONGS_PER_DAY", DEFAULT_PER_DAY) if per_day is None else per_day
        self.per_hour = (_env_int("KR_SONGS_PER_HOUR_PER_IP", DEFAULT_PER_HOUR_PER_IP)
                         if per_hour_per_ip is None else per_hour_per_ip)
        self.clock = clock
        self._lock = threading.Lock()
        self._day = None
        self._today = 0
        self._by_ip: dict[str, deque] = {}

    def _roll(self, now: float) -> None:
        day = datetime.fromtimestamp(now, KST).date()
        if day != self._day:
            # 날이 바뀌면 하루 횟수를 비우고, 쌓인 IP 기록도 한 시간 안의 것만 남긴다
            self._day, self._today = day, 0
            self._by_ip = {ip: q for ip, q in self._by_ip.items() if q and now - q[-1] < 3600}

    def try_take(self, ip: str) -> str | None:
        """한 곡을 더 만들어도 되면 세고 None, 안 되면 앱에 보여 줄 이유를 돌려준다."""
        now = self.clock()
        with self._lock:
            self._roll(now)
            hits = self._by_ip.setdefault(ip, deque())
            while hits and now - hits[0] >= 3600:
                hits.popleft()
            if self.per_day and self._today >= self.per_day:
                return f"오늘 만들 수 있는 노래({self.per_day}곡)를 다 만들었어요. 내일 다시 만들어 주세요"
            if self.per_hour and len(hits) >= self.per_hour:
                return f"한 시간에 {self.per_hour}곡까지 만들 수 있어요. 잠시 후 다시 만들어 주세요"
            self._today += 1
            hits.append(now)
            return None

    def status(self) -> dict:
        with self._lock:
            self._roll(self.clock())
            return {"songsPerDay": self.per_day, "songsPerHourPerIp": self.per_hour, "songsToday": self._today}
