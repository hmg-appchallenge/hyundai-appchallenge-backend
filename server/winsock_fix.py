"""일부 Windows PC에서 소켓 bind()·listen()이 가끔 WinError 10014로 실패하는 문제를 피한다.

은행·인증서용 보안 프로그램(네트워크를 감시하는 드라이버)이 여러 개 깔린 PC에서 이런 일이 생긴다.
실패는 띄엄띄엄 일어나고, 새 소켓으로 다시 하면 된다. 서버가 소켓을 여는 곳은 두 군데라 둘 다 다시 시도하게 한다.
- 접속을 받는 서버 소켓: open_server_socket()
- asyncio가 처음에 만드는 소켓 한 쌍(socket.socketpair): install()이 바꿔 끼운다
Windows가 아니면 아무것도 바꾸지 않는다.
"""
from __future__ import annotations

import socket
import sys
import time

WSAEFAULT = 10014
ATTEMPTS = 40
WAIT_SEC = 0.05

_installed = False
ORIGINAL_LISTEN = None  # 고치기 전 listen (시험 도구가 씀)
_noted = [False]


def _is_flaky(e: OSError) -> bool:
    return getattr(e, "winerror", None) == WSAEFAULT


def _note(log, what: str, attempt: int) -> None:
    if not _noted[0]:
        _noted[0] = True
        log(f"[알림] {what}이(가) WinError {WSAEFAULT}로 실패해서 다시 시도했어 ({attempt}번 만에 성공). "
            "보안 프로그램이 소켓을 감시하는 PC에서 생기는 일이라 서버는 그대로 동작해")


def install(log=print) -> None:
    """listen()을 같은 소켓으로 몇 번 다시 하게 하고, socketpair()는 새 소켓으로 다시 만들게 한다."""
    global _installed, ORIGINAL_LISTEN
    if _installed or sys.platform != "win32":
        return
    original_listen = socket.socket.listen
    original_socketpair = socket.socketpair
    ORIGINAL_LISTEN = original_listen

    def listen(self, *args):
        for attempt in range(5):
            try:
                result = original_listen(self, *args)
                if attempt:
                    _note(log, "listen", attempt + 1)
                return result
            except OSError as e:
                if not _is_flaky(e) or attempt == 4:
                    raise
                time.sleep(WAIT_SEC)

    def socketpair(*args, **kwargs):
        for attempt in range(ATTEMPTS):
            try:
                pair = original_socketpair(*args, **kwargs)
                if attempt:
                    _note(log, "socketpair", attempt + 1)
                return pair
            except OSError as e:
                if not _is_flaky(e) or attempt == ATTEMPTS - 1:
                    raise
                time.sleep(WAIT_SEC)

    socket.socket.listen = listen
    socket.socketpair = socketpair
    _installed = True


def open_server_socket(host: str, port: int, log=print, backlog: int = 128) -> socket.socket:
    """접속을 받는 소켓을 연다. WinError 10014면 새 소켓으로 다시 한다 (포트가 이미 쓰이는 등 다른 오류는 그대로 낸다)."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, socktype, proto, _, addr = infos[0]
    last: OSError | None = None
    for attempt in range(ATTEMPTS):
        sock = socket.socket(family, socktype, proto)
        try:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(addr)
            sock.listen(backlog)
            sock.set_inheritable(True)
            if attempt:
                _note(log, "서버 소켓 열기", attempt + 1)
            return sock
        except OSError as e:
            sock.close()
            last = e
            if not _is_flaky(e):
                raise
            time.sleep(WAIT_SEC)
    assert last is not None
    raise last
