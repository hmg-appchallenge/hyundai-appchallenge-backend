"""python -m server [--mode real|mock] [--port 8000] [--log 파일] [--download]"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import threading
from pathlib import Path

from .config import KEYS_FILE, MODELS_DIR, PID_FILE


class Log:
    def __init__(self, path: Path | None):
        self.f = None
        self.lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        with self.lock:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)
            if self.f:
                self.f.write(line + "\n")
                self.f.flush()


def load_keys(path: Path) -> list[str]:
    """ai-keys.env의 KEY=값을 환경 변수로 올린다. 값은 화면이나 기록에 남기지 않는다."""
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v
            loaded.append(k)
    return loaded


def main(argv: list[str] | None = None) -> int:
    for k, v in {"KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                 "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                 "SUPERTONIC_CACHE_DIR": str(MODELS_DIR / "supertonic3")}.items():
        os.environ.setdefault(k, v)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="server", description="차량 리듬 동요 노래 생성 서버")
    ap.add_argument("--mode", choices=["real", "mock"], default="real")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default=None)
    ap.add_argument("--download", action="store_true", help="화자·VAD 모델과 AI 모델을 내려받고 끝낸다")
    ap.add_argument("--stt-model", default="large-v3-turbo")
    ap.add_argument("--voice", default="F1")
    ap.add_argument("--tts-steps", type=int, default=5, help="TTS 계산 단계 (5: 빠름, 10: 조금 더 고움)")
    args = ap.parse_args(argv)
    log = Log(Path(args.log) if args.log else None)
    from .winsock_fix import install as fix_winsock

    fix_winsock(log)
    loaded = load_keys(KEYS_FILE)
    if loaded:
        log(f"키 불러옴: {', '.join(loaded)}")

    if args.download:
        from .download import download_all

        return download_all(args.stt_model, log)

    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    except OSError:
        pass

    from .engines import MockEngine, RealEngine
    from .jobs import Store
    from .app import create_app

    engine = MockEngine(log) if args.mode == "mock" else RealEngine(log, stt_model=args.stt_model, voice=args.voice, tts_steps=args.tts_steps)
    store = Store(engine, log)
    api_key = os.environ.get("KR_API_KEY") or None
    app = create_app(store, api_key, log)
    log(f"노래 생성 서버 ({args.mode}) http://{args.host}:{args.port}  (에뮬레이터 앱에서는 http://10.0.2.2:{args.port})")
    log(f"화자 모델: {engine.encoder.name} / 인증 키: {'사용 (같은 PC 에뮬레이터는 키 없이 받아요)' if api_key else '없음'}")
    import logging

    import uvicorn

    class ToLog(logging.Handler):
        """uvicorn이 내는 안내·오류(접속 받는 주소, 포트를 못 여는 경우 등)도 로그 파일에 남긴다."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                log(f"[uvicorn] {record.getMessage()}")
            except Exception:
                pass

    # Config가 uvicorn 로그 설정을 새로 하므로, 만든 뒤에 로그 파일 쪽 핸들러를 붙인다
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info", access_log=False)
    logging.getLogger("uvicorn.error").addHandler(ToLog(logging.INFO))
    server = uvicorn.Server(config)
    try:
        if sys.platform == "win32":
            # 보안 프로그램이 깔린 PC에서는 소켓 열기가 가끔 WinError 10014로 실패한다 (server/winsock_fix.py).
            # 서버 소켓을 다시 시도하며 직접 열어 넘기고, 단순한 select 방식 루프로 돌린다
            import asyncio

            from .winsock_fix import open_server_socket

            sock = open_server_socket(args.host, args.port, log)
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(server.serve(sockets=[sock]))
            finally:
                loop.close()
                sock.close()
        else:
            server.run()
    except BaseException as e:
        import traceback

        log(f"[멈춤] 서버가 꺼졌어: {type(e).__name__}: {e}")
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            log(traceback.format_exc())
        raise
    finally:
        try:
            if PID_FILE.exists() and PID_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
