"""켜져 있는 서버에 앱처럼 요청을 보내 전체 흐름을 확인한다.

python -m server.smoketest [--base http://127.0.0.1:8000] [--session 파일.wav] [--enroll a.wav,b.wav] [--out 결과.mp3]
녹음 파일을 안 주면 agent/files 의 test-session.wav, test-enroll-*.wav 를 쓰고, 그것도 없으면 삐 소리로 대신한다.
--base가 이 PC(127.0.0.1, localhost)가 아니면 ai-keys.env의 KR_API_KEY를 X-Api-Key 헤더로 붙인다 (값은 화면에 안 보여 줌).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

from .config import ROOT

FILES = ROOT / "agent" / "files"
API_KEY = ""  # 바깥 주소로 시험할 때만 채운다


def call(method: str, url: str, body: bytes | None = None, ctype: str | None = None, timeout: float = 60):
    # 앱처럼 연결 오류는 몇 번 다시 시도한다 (보안 프로그램이 많은 PC에서 WinError 10014가 가끔 난다)
    for attempt in range(5):
        req = urllib.request.Request(url, data=body, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        if API_KEY:
            req.add_header("X-Api-Key", API_KEY)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return r.status, data
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, ConnectionError) as e:
            if attempt == 4:
                raise
            print(f"   (연결 오류, 다시 시도: {getattr(e, 'reason', e)})")
            time.sleep(0.3 * (attempt + 1))


def jcall(method: str, url: str, obj=None, timeout: float = 60):
    body = json.dumps(obj).encode("utf-8") if obj is not None else None
    status, data = call(method, url, body, "application/json" if body is not None else None, timeout)
    try:
        return status, json.loads(data.decode("utf-8")) if data else None
    except json.JSONDecodeError:
        return status, data[:200]


def multipart(fields: dict[str, tuple[str, bytes, str] | str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):
            fname, data, ctype = value
            out.write(f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                      f"Content-Type: {ctype}\r\n\r\n".encode())
            out.write(data)
        else:
            out.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode("utf-8"))
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def beeps(seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(seconds * 16000)) / 16000
    x = 0.2 * np.sin(2 * np.pi * 330 * t) * ((t % 1.5) < 0.5)
    return x.astype(np.float32)


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--session", default=None)
    ap.add_argument("--enroll", default=None)
    ap.add_argument("--out", default=str(ROOT / "agent" / "outbox" / "smoketest.mp3"))
    ap.add_argument("--realtime", action="store_true", help="2초 조각을 실제 시간처럼 2초마다 보낸다")
    ap.add_argument("--no-song", action="store_true", help="조각만 보내고 완성하기는 부르지 않는다 (노래 생성 비용 없이 화자·단어만 확인)")
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    t_start = time.time()
    if not base.startswith(("http://127.0.0.1", "http://localhost")):
        import os

        from .__main__ import load_keys
        from .config import KEYS_FILE

        global API_KEY
        load_keys(KEYS_FILE)
        API_KEY = os.environ.get("KR_API_KEY", "").strip()
        print("팀 키:", "붙임" if API_KEY else "없음 (ai-keys.env에 KR_API_KEY가 없어요)")

    print("1) 서버 상태")
    for _ in range(120):
        try:
            st, h = jcall("GET", base + "/v1/health", timeout=5)
            if st == 200 and h.get("ready"):
                break
            print("   준비 중...", h)
        except Exception as e:
            print("   연결 안 됨:", type(e).__name__)
        time.sleep(2)
    print("  ", h)

    enroll_files = [Path(p) for p in args.enroll.split(",")] if args.enroll else sorted(FILES.glob("test-enroll-*.wav"))
    embeddings = []
    print("2) 목소리 등록")
    for f in enroll_files:
        body, ctype = multipart({"audio": (f.name, f.read_bytes(), "audio/wav")})
        st, data = call("POST", base + "/v1/voices/embed", body, ctype)
        res = json.loads(data.decode("utf-8"))
        emb = res.get("embedding")
        print(f"   {f.name}: {st} quality={res.get('quality')} speechSec={res.get('speechSec')} dim={len(emb) if emb else 0}")
        embeddings.append(emb if res.get("quality") == "ok" else None)
    # 등록 안 한 가족 한 명도 넣는다
    members = [{"memberId": i + 1, "instrument": inst, "embedding": embeddings[i] if i < len(embeddings) else None}
               for i, inst in enumerate(["DRUM", "SHAKER", "BELL"])]

    print("3) 녹음 세션")
    st, s = jcall("POST", base + "/v1/sessions", {"members": members})
    print("  ", st, s)
    sid = s["sessionId"]
    session_file = Path(args.session) if args.session else FILES / "test-session.wav"
    audio = read_wav16k(session_file) if session_file.exists() else beeps()
    print(f"   녹음 {len(audio) / 16000:.1f}초 ({session_file.name if session_file.exists() else '삐 소리'})")
    step = 32000
    heard = []
    for seq, i in enumerate(range(0, len(audio), step)):
        status, data = call("POST", f"{base}/v1/sessions/{sid}/chunks?seq={seq}", pcm(audio[i:i + step]),
                            "application/octet-stream")
        r = json.loads(data.decode("utf-8"))
        heard += [w["text"] for w in r.get("newWords", [])]
        print(f"   seq {seq:>2}: {status} speaking={r.get('speaking')} newWords={[(w['text'], w['memberId']) for w in r.get('newWords', [])]}")
        if args.realtime:
            time.sleep(2.0)
    time.sleep(0.5)
    st, dup = jcall("POST", f"{base}/v1/sessions/{sid}/chunks?seq=0", None)
    print("   같은 seq 다시 보내기:", st)
    if args.no_song:
        st, _ = jcall("DELETE", f"{base}/v1/sessions/{sid}")
        print(f"   완성하기 없이 세션 지움: {st}")
        print(f"끝 ({time.time() - t_start:.1f}초)")
        return 0

    print("4) 완성하기")
    drive = {"speedKmh": [42, 41, 40, 38, 30, 20, 0, 0, 0, 12, 25, 38, 45, 52], "intervalSec": 1.0,
             "outsideTempC": 14.5, "startedAt": "2026-11-11T13:40:00+09:00"}
    t0 = time.time()
    st, job = jcall("POST", f"{base}/v1/sessions/{sid}/finish", {"drive": drive, "beat": {"bpm": 110, "beatType": 8}})
    print("  ", st, job)
    st2, again = jcall("POST", f"{base}/v1/sessions/{sid}/finish", {"drive": drive})
    print("   완성하기 두 번:", st2, again)
    song = job["songId"]
    last = None
    res = None
    while time.time() - t0 < 300:
        st, res = jcall("GET", f"{base}/v1/songs/{song}")
        key = (res["status"], res["step"])
        if key != last:
            print(f"   {time.time() - t0:5.1f}초 {res['status']} {res['step']} {res['progress']}")
            last = key
        if res["status"] in ("DONE", "FAILED", "CANCELED"):
            break
        time.sleep(0.5)
    if res["status"] != "DONE":
        print("   실패:", res.get("error"))
        return 1
    r = res["result"]
    print(f"5) 결과: '{r['title']}' {r['durationMs'] / 1000:.1f}초, BPM {r['bpm']}, {r['beatType']}비트 (만드는 데 {time.time() - t0:.1f}초)")
    for ln in r["lyrics"]:
        marks = ", ".join(f"{ln['text'][h['start']:h['end']]}({h['kind'][0]}{h['memberId']})" for h in ln["highlights"])
        print(f"   {ln['startMs'] / 1000:6.2f}~{ln['endMs'] / 1000:6.2f}  [{ln['section']:<6}] {ln['text']}   {marks}")
    print("   words:", r["words"])
    print("   onomatopoeia:", r["onomatopoeia"])
    print("   members:", r["members"])
    print("   engine:", r.get("engine"))
    status, mp3 = call("GET", base + r["audio"]["url"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(mp3)
    print(f"6) mp3 받기: {status}, {len(mp3) / 1e6:.2f} MB -> {out}")
    st, _ = call("DELETE", f"{base}/v1/songs/{song}")
    st2, gone = jcall("GET", f"{base}/v1/songs/{song}")
    print(f"7) 지우기: {st}, 다시 조회 {st2}")
    st, nf = jcall("POST", f"{base}/v1/sessions/nope/chunks?seq=0", None)
    print(f"   없는 세션에 조각: {st}")
    print(f"끝 ({time.time() - t_start:.1f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
