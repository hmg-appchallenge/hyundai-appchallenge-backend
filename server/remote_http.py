"""외부 API(ElevenLabs, Supabase)를 부를 때 쓰는 작은 HTTP 도우미 (표준 라이브러리만 씀).

이 PC는 보안 프로그램 때문에 연결을 여는 순간 WinError 10014가 가끔 난다.
그때는 요청이 아직 나가지 않은 것이라 다시 시도해도 돈이 두 번 나가지 않는다.
요청을 보낸 뒤의 오류(응답 대기 시간 초과 등)는 다시 보내지 않는다 (노래 생성이 두 번 될 수 있어서).
"""
from __future__ import annotations

import errno
import json
import time
import urllib.error
import urllib.request

CONNECT_RETRIES = 5


def _not_sent(e: BaseException) -> bool:
    """연결을 열지 못해서 요청이 나가지 않은 오류인지."""
    reason = getattr(e, "reason", e)
    if isinstance(reason, OSError):
        if getattr(reason, "winerror", None) == 10014:
            return True
        if reason.errno in (errno.ECONNREFUSED, errno.EADDRNOTAVAIL, errno.ENETUNREACH):
            return True
        if isinstance(reason, ConnectionRefusedError):
            return True
    return "10014" in str(reason)


def request(method: str, url: str, headers: dict | None = None, body: bytes | None = None,
            timeout: float = 60.0) -> tuple[int, dict, bytes]:
    """(상태 코드, 응답 머리글, 본문). HTTP 오류도 예외 없이 돌려준다."""
    last: BaseException | None = None
    for attempt in range(CONNECT_RETRIES):
        req = urllib.request.Request(url, data=body, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read()
        except (urllib.error.URLError, OSError) as e:
            last = e
            if not _not_sent(e) or attempt == CONNECT_RETRIES - 1:
                raise
            time.sleep(0.4 * (attempt + 1))
    raise last if last else RuntimeError("요청 실패")


def request_json(method: str, url: str, headers: dict | None = None, obj=None,
                 timeout: float = 60.0) -> tuple[int, object]:
    h = dict(headers or {})
    body = None
    if obj is not None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    h.setdefault("Accept", "application/json")
    status, _hdr, data = request(method, url, h, body, timeout)
    try:
        return status, json.loads(data.decode("utf-8")) if data else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, data[:500].decode("utf-8", "replace")
