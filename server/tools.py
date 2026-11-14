"""시험용 도구.

python -m server.tools make-test-audio   : TTS로 아이·엄마 흉내 목소리를 만들어 agent/files 에 넣는다
                                          (test-enroll-a.wav, test-enroll-b.wav, test-session.wav, debug_mic.wav)
python -m server.tools make-test-audio 대본.txt [fast] : 대본으로 <대본 이름>.wav를 만든다 (fast면 줄 사이를 거의 안 쉬고 <이름>-fast.wav)
python -m server.tools check-keys        : ai-keys.env의 Claude·ElevenLabs·Supabase 키가 되는지 본다 (값은 안 보여 줌)
python -m server.tools tunnel-check [주소] : 공개 주소(안 주면 share-server.bat의 임시 주소)로 바깥에서 붙듯이 시험한다 (키 없이 401, 키로 세션 열고 닫기)
python -m server.tools hf-token-save [초] : 클립보드에 복사한 허깅페이스 토큰을 확인해 hf-token.txt로 저장하고 클립보드를 비운다 (초를 주면 그동안 복사를 기다림, 토큰은 화면에 안 보여 줌)
python -m server.tools kr-key-copy [초]   : 팀 키(KR_API_KEY)를 클립보드에 넣고, 초(기본 90)가 지나면 비운다 (카톡으로 팀원에게 보낼 때, 값은 안 보여 줌)
python -m server.tools hf-secrets SPACE  : ai-keys.env의 키 3개를 Space Secrets에, MOCK_GENERATION=0을 Variables에 넣는다 (값은 안 보여 줌)
python -m server.tools hf-pause SPACE    : Space를 멈춘다 (멈춘 동안 요금 안 나감)
python -m server.tools hf-deploy SPACE   : deploy/hf-space.tar를 허깅페이스 Space에 올린다 (토큰은 hf-token.txt, 화면에 안 보여 줌)
python -m server.tools hf-status SPACE   : Space가 빌드 중인지, 켜졌는지 본다
python -m server.tools hf-logs SPACE [build|run] : Space 빌드·실행 기록 끝부분을 본다
python -m server.tools words-probe [파일] [fast] : agent/files 의 녹음을 세션처럼 넣어 들린 말·확신도·가사 단어 여부를 본다
python -m server.tools live-probe 주소 [파일] : 켜진 서버에 녹음을 앱처럼 2초마다 보내 단어가 언제 오는지 잰다 (완성은 안 부르고 세션을 지움)
python -m server.tools bench [모델...]    : 이 PC에서 인식·TTS·노래 한 곡이 얼마나 걸리는지 잰다
python -m server.tools socket-test [heavy] [fix] / socket-stress : Windows 소켓 오류(WinError 10014) 진단
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from .config import AI_DIR, MODELS_DIR, ROOT

if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

FILES = ROOT / "agent" / "files"
ENROLL_LINE = "안녕, 우리 같이 노래하자!"
ENROLL_LINE_2 = "오늘은 신나는 노래를 만들 거야!"  # 등록은 실제로 말한 시간이 3초 넘어야 해서 한 문장 더
# (누가, 말) — A: 아이(F2, 음높이 +4), B: 엄마(F1, 그대로)
SESSION = [("A", "사과!"), ("B", "둥둥 둥둥!"), ("A", "랄랄라~"), ("B", "엄마가 좋아!"), ("A", "강아지!"),
           ("A", "부릉부릉!"), ("B", "우리 바다 가자!"), ("A", "랄랄라~"), ("B", "사과 맛있다!"), ("A", "둥둥!")]
VOICES = {"A": ("F2", 4.0), "B": ("F1", 0.0)}


def write_wav16k(path: Path, x: np.ndarray, sr: int) -> None:
    from songmaker.dsp import resample

    y = resample(x, sr, 16000)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def _adb() -> str:
    """에뮬레이터를 다루는 adb 경로 (ANDROID_HOME, ANDROID_SDK_ROOT, Windows 기본 설치 위치 순서)."""
    import os
    import shutil

    exe = "adb.exe" if os.name == "nt" else "adb"
    for base in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")):
        if base and (Path(base) / "platform-tools" / exe).exists():
            return str(Path(base) / "platform-tools" / exe)
    return shutil.which("adb") or "adb"


def screen_record(action: str, name: str = "demo", seconds: int = 180) -> int:
    """에뮬레이터 화면 녹화 (시연 영상용). start로 켜고, 앱을 조작한 뒤 stop, pull로 agent/outbox에 꺼낸다.
    소리는 녹화되지 않고, 한 번에 최대 180초다."""
    import re
    import subprocess
    import time

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        print("이름은 영문, 숫자, _, - 로만 써요")
        return 2
    adb, remote = _adb(), f"/data/local/tmp/{name}.mp4"

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([adb, "-e", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")

    def recording() -> bool:
        return bool(run("shell", "pidof", "screenrecord").stdout.strip())

    if action == "start":
        run("shell", "pkill", "-INT", "screenrecord")
        sec = max(1, min(int(seconds), 180))
        run("shell", f"setsid nohup screenrecord --bit-rate 16000000 --time-limit {sec} {remote} "
                     "</dev/null >/dev/null 2>&1 &")
        for _ in range(12):
            if recording():
                print(f"녹화 시작: {remote} (최대 {sec}초)")
                return 0
            time.sleep(0.25)
        print("녹화를 시작하지 못했어요")
        return 1
    if action == "stop":
        run("shell", "pkill", "-INT", "screenrecord")
        for _ in range(40):
            if not recording():
                break
            time.sleep(0.25)
        print("녹화 멈춤" if not recording() else "아직 녹화가 안 멈췄어요")
        return 0
    if action == "pull":
        out = ROOT / "agent" / "outbox" / f"{name}.mp4"
        r = run("pull", remote, str(out))
        print((r.stdout or r.stderr).strip().splitlines()[-1:] or "pull 결과 없음")
        return r.returncode
    print("start, stop, pull 중 하나를 써요")
    return 2


def emu_record(action: str, name: str = "demo", seconds: int = 180) -> int:
    """에뮬레이터 바깥(PC)에서 화면을 녹화한다 (adb emu screenrecord, webm). 에뮬레이터 안에서 녹화하는
    screen-record보다 에뮬레이터 부담이 적다. 결과는 agent/outbox/<이름>.webm."""
    import re
    import subprocess

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        print("이름은 영문, 숫자, _, - 로만 써요")
        return 2
    adb = _adb()
    out = ROOT / "agent" / "outbox" / f"{name}.webm"
    if action == "start":
        sec = max(1, min(int(seconds), 600))
        r = subprocess.run([adb, "-e", "emu", "screenrecord", "start", "--time-limit", str(sec),
                            "--bit-rate", "6000000", str(out)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    elif action == "stop":
        r = subprocess.run([adb, "-e", "emu", "screenrecord", "stop"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    else:
        print("start, stop 중 하나를 써요")
        return 2
    msg = (r.stdout + r.stderr).strip().replace("\r", " ").replace("\n", " ")
    print(f"emu-record {action}: {msg or 'OK'} -> {out}")
    return 0 if "KO" not in msg else 1


APP_PKG = "com.hmgappchallenge.kidsrhythm"


def app_prefs(action: str, name: str) -> int:
    """앱 설정 파일(shared_prefs/<이름>.xml)을 agent/outbox로 꺼내거나(get), agent/files/<이름>.xml을 넣는다(put).
    시연 영상용 데이터(가족 이름 등)를 준비할 때 쓴다. put은 앱을 먼저 멈춘다 (켜져 있으면 앱이 덮어써서)."""
    import re
    import subprocess

    if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", name):
        print("이름은 영문, 숫자, _ 로만 써요")
        return 2
    adb = _adb()

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([adb, "-e", *args], capture_output=True)

    user = run("shell", "am", "get-current-user").stdout.decode(errors="replace").strip() or "0"
    if action == "get":
        r = run("exec-out", "run-as", APP_PKG, "--user", user, "cat", f"shared_prefs/{name}.xml")
        out = ROOT / "agent" / "outbox" / f"{name}.xml"
        out.write_bytes(r.stdout)
        print(f"꺼냄: {out} ({len(r.stdout)} bytes, 사용자 {user})")
        return 0 if r.stdout else 1
    if action == "put":
        src = FILES / f"{name}.xml"
        if not src.exists():
            print("파일이 없어요:", src)
            return 1
        run("shell", "am", "force-stop", APP_PKG)
        run("push", str(src), f"/data/local/tmp/{name}.xml")
        r = run("shell", f"cat /data/local/tmp/{name}.xml | run-as {APP_PKG} --user {user} "
                         f"sh -c 'cat > shared_prefs/{name}.xml'")
        check = run("exec-out", "run-as", APP_PKG, "--user", user, "cat", f"shared_prefs/{name}.xml").stdout
        same = check == src.read_bytes()
        print(f"넣음: shared_prefs/{name}.xml (사용자 {user}, 같은지 {same})" + (f" {r.stderr.decode(errors='replace')}" if r.stderr else ""))
        return 0 if same else 1
    print("get, put 중 하나를 써요")
    return 2


def read_script(path: Path) -> list[tuple[str, str]]:
    """대본 파일을 (누가, 말) 목록으로 읽는다. 한 줄에 'A: 엄마 나 배고파!'처럼 쓴다 (A=아이, B=엄마).
    빈 줄과 #으로 시작하는 줄은 건너뛴다."""
    lines = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        who, _, text = s.partition(":")
        who, text = who.strip().upper(), text.strip()
        if who not in VOICES or not text:
            raise ValueError(f"대본 줄을 못 읽었어요: {raw!r} (예: 'A: 엄마 나 배고파!')")
        lines.append((who, text))
    return lines


def make_test_audio(script: Path | None = None, fast: bool = False) -> int:
    """TTS로 시험용 녹음을 만든다. 대본을 주면 그 대본으로 agent/files/<대본 이름>.wav만 만든다
    (목소리 등록 파일이 없으면 그것도 만든다). 대본이 없으면 기본 대본으로 test-session.wav와 debug_mic.wav를 만든다.
    fast면 줄 사이를 0.12초만 쉬어서(말소리 구간이 끊기지 않게) <대본 이름>-fast.wav로 만든다.
    줄마다 말한 시간은 같은 이름의 .json에 남긴다 (live-probe가 단어가 몇 초 늦게 왔는지 셀 때 쓴다)."""
    import json

    from songmaker.dsp import SR
    from songmaker.tts import SupertonicTTS, Voice

    lines = SESSION
    if script is not None:
        script = script if script.is_absolute() or script.exists() else FILES / script
        lines = read_script(script)

    tts = SupertonicTTS("F1", MODELS_DIR, steps=10)

    def voice(who: str) -> Voice:
        name, pitch = VOICES[who]
        tts.use_voice(name)
        return Voice(tts, pitch=pitch)

    gap = lambda s: np.zeros(int(s * SR), np.float32)  # noqa: E731
    for who, fname in (("A", "test-enroll-a.wav"), ("B", "test-enroll-b.wav")):
        if script is not None and (FILES / fname).exists():
            continue
        v = voice(who)
        line = v.say(ENROLL_LINE) * 0.8
        line2 = v.say(ENROLL_LINE_2) * 0.8
        write_wav16k(FILES / fname, np.concatenate([gap(0.5), line, gap(0.6), line2, gap(0.6), line, gap(0.5)]), SR)
        print("만듦:", FILES / fname)
    pause = 0.12 if fast else 0.9
    parts, pos, timing = [gap(0.6)], 0.6, []
    cache: dict[str, Voice] = {}
    for who, text in lines:
        if who not in cache:
            cache[who] = voice(who)
        else:
            tts.use_voice(VOICES[who][0])
        said = cache[who].say(text) * 0.8
        timing.append({"who": who, "text": text, "start": round(pos, 2), "end": round(pos + len(said) / SR, 2)})
        parts += [said, gap(pause)]
        pos += len(said) / SR + pause
    parts.append(gap(1.0))
    session = np.concatenate(parts)
    stem = script.stem + ("-fast" if fast else "") if script is not None else None
    outs = (f"{stem}.wav",) if stem else ("test-session.wav", "debug_mic.wav")
    for fname in outs:
        write_wav16k(FILES / fname, session, SR)
        (FILES / fname).with_suffix(".json").write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
        print("만듦:", FILES / fname, f"({len(session) / SR:.1f}초, {len(lines)}줄)")
    return 0


def bench(models: list[str]) -> int:
    """이 PC에서 음성 인식·TTS·노래 만들기가 얼마나 걸리는지 잰다 (에뮬레이터를 켠 상태에서 재면 실제와 비슷하다)."""
    import time

    from .config import SR_IN

    session = FILES / "test-session.wav"
    if session.exists():
        with wave.open(str(session), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    else:
        audio = np.zeros(SR_IN * 4, np.float32)
    clip = audio[: int(2.5 * SR_IN)]

    from songmaker.stt import Transcriber

    for name in models:
        t0 = time.time()
        try:
            stt = Transcriber(name, MODELS_DIR / "whisper")
        except Exception as e:
            print(f"STT {name}: 불러오기 실패 {type(e).__name__}: {e}")
            continue
        load = time.time() - t0
        times = []
        for beam in (5, 1):
            stt_beam = getattr(stt, "beam_size", None)
            if stt_beam is not None:
                stt.beam_size = beam
            t1 = time.time()
            words = stt.transcribe(clip)
            times.append((beam, time.time() - t1, " ".join(w.text for w in words)))
        print(f"STT {name}: 불러오기 {load:.1f}초 / 2.5초 말 인식: "
              + ", ".join(f"beam{b} {t:.2f}초 '{txt}'" for b, t, txt in times), flush=True)
        del stt

    from songmaker.tts import SupertonicTTS, Voice

    for steps in (10, 5):
        t0 = time.time()
        tts = SupertonicTTS("F1", MODELS_DIR, steps=steps)
        load = time.time() - t0
        t1 = time.time()
        for text in ("사과는 빨갛다", "둥둥 둥둥", "신나는 우리 노래"):
            tts.synth(text)
        synth = (time.time() - t1) / 3
        voice = Voice(tts, pitch=3.0)
        t2 = time.time()
        voice.say("사과는 맛있다")
        say = time.time() - t2
        print(f"TTS steps={steps}: 불러오기 {load:.1f}초 / 한 줄 합성 {synth:.2f}초 / 음높이까지 {say:.2f}초", flush=True)

        from .engines import MockEngine
        from .live import Member

        engine = MockEngine(log=lambda m: None, step_delay=0.0)
        engine.voice = Voice(tts, pitch=3.0)
        members = [Member(1, "DRUM"), Member(2, "SHAKER"), Member(3, "BELL")]
        t3 = time.time()
        engine.make_song(audio, [], members, None, {"bpm": 110, "beatType": 8}, lambda s, p: None, lambda m: None)
        print(f"노래 한 곡 (TTS steps={steps}): {time.time() - t3:.1f}초", flush=True)
    return 0


def socket_stress(n: int = 300) -> int:
    """listen() 실패가 얼마나 자주 나는지, 같은 소켓으로 다시 하면 되는지 센다."""
    import socket
    import time

    from . import winsock_fix

    fails = same_ok = same_stuck = 0
    attempts_needed = []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        try:
            winsock_fix.ORIGINAL_LISTEN(s, 16) if winsock_fix.ORIGINAL_LISTEN else s.listen(16)
        except OSError:
            fails += 1
            for k in range(1, 41):
                time.sleep(0.05)
                try:
                    (winsock_fix.ORIGINAL_LISTEN or socket.socket.listen)(s, 16)
                    same_ok += 1
                    attempts_needed.append(k)
                    break
                except OSError:
                    continue
            else:
                same_stuck += 1
        finally:
            s.close()
    print(f"listen 시도 {n}번: 실패 {fails}번, 같은 소켓으로 다시 해서 성공 {same_ok}번 (필요 횟수 {attempts_needed[:20]}), "
          f"2초 동안 안 된 것 {same_stuck}번")
    return 0


def socket_test(heavy: bool = False) -> int:
    """이 PC에서 서버가 접속을 받을 수 있는 주소·포트를 찾는다 (bind + listen).
    heavy=True면 서버가 쓰는 AI 라이브러리를 먼저 불러온 뒤에 시험한다."""
    import asyncio
    import socket
    import traceback

    if heavy:
        for mod in ("numpy", "sherpa_onnx", "onnxruntime", "ctranslate2", "faster_whisper", "kiwipiepy"):
            try:
                __import__(mod)
                print(f"loaded {mod}")
            except Exception as e:
                print(f"skip  {mod}: {type(e).__name__}: {e}")
        try:
            a, b = socket.socketpair()
            a.close()
            b.close()
            print("OK    socketpair after imports")
        except OSError as e:
            print(f"FAIL  socketpair after imports: {e}")
            traceback.print_exc()
        try:
            loop = asyncio.SelectorEventLoop()
            print("OK    SelectorEventLoop() after imports")
            loop.close()
        except OSError as e:
            print(f"FAIL  SelectorEventLoop() after imports: {e}")
            traceback.print_exc()

    cases = [("127.0.0.1", 8000), ("127.0.0.1", 0), ("127.0.0.1", 18000), ("0.0.0.0", 8000), ("0.0.0.0", 0),
             ("::1", 8000), ("::1", 0), ("::", 0), ("localhost", 8000)]
    ok_any = False
    for host, port in cases:
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        step = "bind"
        try:
            s.bind((host, port))
            step = "listen"
            s.listen(16)
            print(f"OK    {host}:{port} -> {s.getsockname()[:2]}")
            ok_any = True
        except OSError as e:
            print(f"FAIL  {host}:{port} ({step}) {type(e).__name__}: {e}")
        finally:
            s.close()

    # asyncio가 처음에 만드는 '자기 자신에게 연결한 소켓 한 쌍'을 단계별로 따라 해 본다
    lsock = socket.socket()
    step = "bind"
    try:
        lsock.bind(("127.0.0.1", 0))
        step = "listen"
        lsock.listen()
        addr = lsock.getsockname()[:2]
        csock = socket.socket()
        step = "connect"
        csock.connect(addr)
        step = "accept"
        ssock, peer = lsock.accept()
        step = "getsockname(client)"
        cname = csock.getsockname()
        step = "getpeername(server side)"
        pname = ssock.getpeername()
        print(f"OK    loopback pair: client={cname} peer={pname} accept_peer={peer}")
        step = "send/recv"
        csock.sendall(b"x")
        print(f"OK    loopback send/recv: {ssock.recv(1)!r}")
        ssock.close()
        csock.close()
    except OSError as e:
        print(f"FAIL  loopback pair at {step}: {type(e).__name__}: {e}")
    finally:
        lsock.close()
    try:
        a, b = socket.socketpair()
        print("OK    socket.socketpair()")
        a.close()
        b.close()
    except OSError as e:
        print(f"FAIL  socket.socketpair(): {type(e).__name__}: {e}")

    async def try_asyncio(host: str, port: int) -> None:
        server = await asyncio.start_server(lambda r, w: None, host, port)
        print(f"OK    asyncio {host}:{port}")
        server.close()
        await server.wait_closed()

    for name, factory, port in [("selector", asyncio.SelectorEventLoop, 8001),
                                ("proactor", getattr(asyncio, "ProactorEventLoop", None), 8002)]:
        if factory is None:
            continue
        try:
            loop = factory()
        except OSError as e:
            print(f"FAIL  asyncio {name} loop: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        try:
            loop.run_until_complete(try_asyncio("127.0.0.1", port))
            print(f"OK    asyncio {name} loop")
        except OSError as e:
            print(f"FAIL  asyncio {name} serve: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            loop.close()
    return 0 if ok_any else 1


def words_probe(path: Path, realtime: bool = True) -> int:
    """녹음 파일을 앱처럼 2초씩 세션에 넣어, 구간마다 들린 말과 확신도, 가사 단어로 쓰는지(O/X) 보여 준다."""
    import threading
    import time

    from kiwipiepy import Kiwi
    from songmaker.stt import Transcriber
    from songmaker.text import analyze, usable_noun

    from .engines import BaseEngine, RealEngine
    from .live import LiveSession, Member
    from .stream_asr import StreamAsr

    class ProbeEngine(RealEngine):  # TTS 없이 인식에 필요한 것만 불러온다
        def __init__(self):
            BaseEngine.__init__(self, log=print)
            self.kiwi = Kiwi()
            self.stt = Transcriber("large-v3-turbo", MODELS_DIR / "whisper")
            self.stream_asr = StreamAsr(log=print)
            print("실시간 받아쓰기:", self.stream_asr.name)
            self._stt_lock = threading.Lock()
            self.ready.set()

    with wave.open(str(path), "rb") as w:
        pcm = w.readframes(w.getnframes())
    engine = ProbeEngine()
    sess = LiveSession("probe", [Member(1, "DRUM")], engine, log=print)
    step = 2 * 16000 * 2
    t0 = time.time()
    for seq, off in enumerate(range(0, len(pcm), step)):
        r = sess.add_chunk(seq, pcm[off:off + step])
        if r["newWords"]:
            print(f"[{time.time() - t0:5.1f}초] 들린 단어: " + ", ".join(w["text"] for w in r["newWords"]), flush=True)
        if realtime:
            time.sleep(2.0)
    utts = sess.finalize(timeout=180)
    print(f"구간 {len(utts)}개 (전체 {time.time() - t0:.1f}초)")
    for u in utts:
        marks = []
        for w in u.words:
            nouns = [t.form for t in engine.kiwi.tokenize(w.text) if t.tag in ("NNG", "NNP")]
            verdict = "".join(f" {n}:{'O' if usable_noun(n, w.prob) else 'X'}" for n in nouns)
            marks.append(f"{w.text}({w.prob:.2f}{verdict})")
        print(f"{u.start:6.1f}~{u.end:6.1f}초  " + " ".join(marks), flush=True)
    an = analyze([w for u in utts for w in u.words], "", engine.kiwi)
    print("가사 단어:", ", ".join(dict.fromkeys(n.text for n in an.nouns)) or "없음")
    print("의성어:", ", ".join(dict.fromkeys(o.text for o in an.onos)) or "없음")
    return 0


def live_probe(base: str, path: Path) -> int:
    """켜진 서버에 녹음 파일을 앱처럼 보내(2초 녹음이 끝날 때마다 조각 하나) 단어가 언제 오는지 잰다.
    완성하기는 부르지 않고 세션을 지워서 노래 한도와 요금을 쓰지 않는다.
    같은 이름의 .json(make-test-audio가 남긴 줄별 시간)이 있으면 그 말을 끝내고 몇 초 뒤에 왔는지도 센다."""
    import json
    import os
    import time

    from . import smoketest

    base = base.rstrip("/")
    if not path.exists():
        print(f"녹음 파일이 없어요: {path}")
        return 1
    if not base.startswith(("http://127.0.0.1", "http://localhost")):
        from .__main__ import load_keys
        from .config import KEYS_FILE

        load_keys(KEYS_FILE)
        smoketest.API_KEY = os.environ.get("KR_API_KEY", "").strip()
        print("팀 키:", "붙임" if smoketest.API_KEY else "없음 (ai-keys.env에 KR_API_KEY가 없어요)")
    st, h = smoketest.jcall("GET", base + "/v1/health", timeout=20)
    print("서버:", st, {k: h.get(k) for k in ("version", "mode", "ready", "vad", "streamAsr")} if isinstance(h, dict) else h)
    side = path.with_suffix(".json")
    lines = json.loads(side.read_text(encoding="utf-8")) if side.exists() else []
    speech = smoketest.read_wav16k(path)
    audio = np.concatenate([speech, np.zeros(16000 * 6, np.float32)])  # 끝 단어까지 받으려고 조용한 6초를 더 보낸다
    members = [{"memberId": 1, "instrument": "DRUM", "embedding": None},
               {"memberId": 2, "instrument": "SHAKER", "embedding": None}]
    st, s = smoketest.jcall("POST", base + "/v1/sessions", {"members": members})
    if st != 201:
        print("세션을 못 열었어요:", st, s)
        return 1
    sid = s["sessionId"]
    print(f"녹음 {len(speech) / 16000:.1f}초 ({path.name}) + 조용한 6초, 2초마다 한 조각")
    step = 32000
    arrivals: list[tuple[str, float]] = []
    t0 = time.time()
    try:
        for seq, i in enumerate(range(0, len(audio), step)):
            time.sleep(max(0.0, t0 + (seq + 1) * 2.0 - time.time()))  # 앱처럼 2초 녹음이 끝나면 보낸다
            sent = time.time() - t0
            status, data = smoketest.call("POST", f"{base}/v1/sessions/{sid}/chunks?seq={seq}",
                                          smoketest.pcm(audio[i:i + step]), "application/octet-stream")
            now = time.time() - t0
            if status != 200:
                print(f"   조각 {seq}: {status} {data[:200]!r}")
                continue
            r = json.loads(data.decode("utf-8"))
            words = [w["text"] for w in r.get("newWords", [])]
            arrivals += [(w, now) for w in words]
            print(f"[{now:5.1f}초] 조각 {seq:>2} (응답 {now - sent:.2f}초) 말한 사람={r.get('speaking')} 새 단어={words}",
                  flush=True)
    finally:
        st, _ = smoketest.call("DELETE", f"{base}/v1/sessions/{sid}")
        print("세션 지움:", st)
    if arrivals and lines:
        print("단어가 온 때 (그 말을 끝낸 시각과 비교):")
        delays = []
        for word, at in arrivals:
            said = next((ln for ln in lines if word in ln["text"].replace(" ", "")), None)
            if said is None:
                print(f"   {word}: {at:.1f}초 (대본에서 못 찾음)")
                continue
            delays.append(at - said["end"])
            print(f"   {word}: {at:.1f}초 ('{said['text']}' {said['start']:.1f}~{said['end']:.1f}초, {at - said['end']:+.1f}초)")
        if delays:
            print(f"말을 끝내고 평균 {np.mean(delays):.1f}초, 가장 늦게 {max(delays):.1f}초 뒤에 옴 ({len(delays)}개)")
    print(f"받은 단어 {len(arrivals)}개: {', '.join(w for w, _ in arrivals) or '없음'}")
    return 0


def check_keys() -> int:
    """ai-keys.env의 외부 서비스 키가 실제로 되는지 본다. 키 값은 화면에 남기지 않는다."""
    import os

    from .__main__ import load_keys
    from .config import KEYS_FILE
    from .remote_http import request_json

    loaded = load_keys(KEYS_FILE)
    print("불러온 키 이름:", ", ".join(loaded) or "없음")
    print("MOCK_GENERATION:", os.environ.get("MOCK_GENERATION", "(없음)"))
    ok = True

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Claude: 키 없음")
        ok = False
    else:
        from .remote_song import claude_message

        try:
            text, model = claude_message("한 단어로만 답해: 사과는 무슨 색?", system=None, max_tokens=10,
                                         model="claude-haiku-5-5", timeout=40)
            print(f"Claude: OK ({model}, 답: {text.strip()[:20]})")
        except Exception as e:
            print(f"Claude: 실패 {type(e).__name__}: {str(e)[:300]}")
            ok = False

    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        print("ElevenLabs: 키 없음")
        ok = False
    else:
        status, data = request_json("GET", "https://api.elevenlabs.io/v1/user/subscription", headers={"xi-api-key": key})
        if status == 200 and isinstance(data, dict):
            fields = ["tier", "status", "character_count", "character_limit", "next_character_count_reset_unix"]
            print("ElevenLabs: OK " + ", ".join(f"{k}={data.get(k)}" for k in fields if k in data))
        else:
            print(f"ElevenLabs: 실패 HTTP {status} {str(data)[:300]}")
            ok = False

    url, key = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        print("Supabase: 주소나 키 없음")
    else:
        status, data = request_json("GET", url + "/rest/v1/", headers={"apikey": key})
        if status == 200 and isinstance(data, dict):
            tables = sorted(p.strip("/") for p in (data.get("paths") or {}) if p.strip("/") and "rpc/" not in p)
            print(f"Supabase: OK (테이블 {len(tables)}개: {', '.join(tables) or '아직 없음'})")
            defs = data.get("definitions") or {}
            for t in tables:
                props = (defs.get(t) or {}).get("properties") or {}
                cols = [f"{c}:{(p or {}).get('format') or (p or {}).get('type')}" for c, p in props.items()]
                print(f"  {t}: {', '.join(cols)}")
        else:
            print(f"Supabase: 실패 HTTP {status} {str(data)[:300]}")
    return 0 if ok else 1


def tunnel_check(base: str | None = None) -> int:
    """공개 주소(안 주면 server-url.txt)로 바깥 요청처럼 붙어 본다. 팀 키 값은 화면에 남기지 않는다."""
    import os

    from .__main__ import load_keys
    from .config import KEYS_FILE
    from .remote_http import request_json

    load_keys(KEYS_FILE)
    if not base:
        url_file = ROOT / "server-url.txt"
        if not url_file.exists():
            print("server-url.txt가 없어요. share-server.bat을 먼저 켜 주세요")
            return 1
        base = url_file.read_text(encoding="ascii").strip()
    base = base.rstrip("/")
    key = os.environ.get("KR_API_KEY", "")
    print("주소:", base)
    status, data = request_json("GET", base + "/v1/health", timeout=20)
    print(f"health: HTTP {status} auth={(data or {}).get('auth') if isinstance(data, dict) else '?'}")
    body = {"members": [{"memberId": 1, "instrument": "DRUM", "embedding": None}]}
    status, _ = request_json("POST", base + "/v1/sessions", obj=body, timeout=20)
    print(f"키 없이 세션 열기: HTTP {status} (401이어야 해요)")
    ok = status == 401
    if not key:
        print("팀 키(KR_API_KEY)가 없어요")
        return 1
    status, data = request_json("POST", base + "/v1/sessions", headers={"x-api-key": key}, obj=body, timeout=20)
    print(f"키로 세션 열기: HTTP {status} (201이어야 해요)")
    ok = ok and status == 201
    sid = data.get("sessionId") if isinstance(data, dict) else None
    if sid:
        status, _ = request_json("DELETE", f"{base}/v1/sessions/{sid}", headers={"x-api-key": key}, timeout=20)
        print(f"세션 닫기: HTTP {status} (204여야 해요)")
        ok = ok and status == 204
    print("결과:", "바깥에서 쓸 수 있어요" if ok else "확인이 필요해요")
    return 0 if ok else 1


def _hf_token() -> str | None:
    f = ROOT / "hf-token.txt"
    if not f.exists():
        print("hf-token.txt가 없어요. 허깅페이스 토큰을 kidsrhythm-dev 폴더에 hf-token.txt로 저장해 주세요")
        return None
    return f.read_text(encoding="utf-8-sig").strip()


def _hf_space(api, space: str) -> str:
    return space if "/" in space else f"{api.whoami()['name']}/{space}"


def _http_status(e: Exception) -> str:
    code = getattr(getattr(e, "response", None), "status_code", None)
    return f"HTTP {code}" if code else ""


def _clipboard(clear: bool = False) -> str:
    """Windows 클립보드의 글자를 읽는다. clear=True면 비우기만 한다."""
    import ctypes
    import time
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    for _ in range(20):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.1)
    else:
        raise OSError("클립보드를 열 수 없어요")
    try:
        if clear:
            user32.EmptyClipboard()
            return ""
        handle = user32.GetClipboardData(13)  # CF_UNICODETEXT
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _clipboard_seq() -> int:
    """클립보드가 바뀔 때마다 1씩 느는 번호 (클립보드를 열지 않고 바뀜만 본다)."""
    import ctypes

    user32 = ctypes.WinDLL("user32")
    user32.GetClipboardSequenceNumber.argtypes = []
    user32.GetClipboardSequenceNumber.restype = ctypes.c_uint32
    return int(user32.GetClipboardSequenceNumber())


def hf_token_save(wait_sec: float = 0.0) -> int:
    """클립보드에 복사한 허깅페이스 토큰을 확인해 hf-token.txt로 저장하고 클립보드를 비운다. 토큰 값은 화면에 남기지 않는다.

    wait_sec를 주면 그동안 클립보드가 바뀌기를 기다렸다가 토큰이 들어오는 순간 저장한다.
    """
    import re
    import time

    from huggingface_hub import HfApi

    deadline = time.time() + wait_sec
    while True:
        seq = _clipboard_seq()
        text = _clipboard()
        found = sorted(set(re.findall(r"hf_[A-Za-z0-9]{30,}", text)))
        if len(found) == 1 or time.time() >= deadline:
            break
        while _clipboard_seq() == seq and time.time() < deadline:
            time.sleep(0.3)
    if len(found) != 1:
        # 토큰 값은 빼고 모양만 알려 준다
        odd = sum(1 for c in text.strip() if not c.isalnum() and c != "_")
        print(f"클립보드에 허깅페이스 토큰(hf_로 시작)이 없어요. 토큰 옆 Copy를 다시 눌러 주세요"
              f" (지금 클립보드: 글자 {len(text.strip())}개, hf_ 시작 {text.strip().startswith('hf_')}, 토큰 모양 {len(found)}개, 기호 {odd}개)")
        return 1
    token = found[0]
    try:
        me = HfApi(token=token).whoami()
    except Exception as e:
        print("허깅페이스가 이 토큰을 받지 않았어요:", type(e).__name__, _http_status(e))
        return 1
    (ROOT / "hf-token.txt").write_text(token + "\n", encoding="utf-8")
    _clipboard(clear=True)
    access = (me.get("auth") or {}).get("accessToken") or {}
    print(f"저장함: hf-token.txt (계정 {me.get('name')}, 토큰 이름 {access.get('displayName')}, 종류 {access.get('role')})")
    fine = access.get("fineGrained") or {}
    for item in fine.get("scoped") or []:
        ent = item.get("entity") or {}
        print(f"  {ent.get('type')} {ent.get('name')}: {', '.join(item.get('permissions') or [])}")
    if fine.get("global"):
        print("  계정 전체:", ", ".join(fine.get("global") or []))
    print("클립보드는 비웠어요")
    return 0



def kr_key_copy(keep_sec: float = 90.0) -> int:
    """팀 키(KR_API_KEY)를 클립보드에 넣는다. keep_sec가 지나도 그대로면 비운다. 키 값은 화면에 남기지 않는다."""
    import os
    import subprocess
    import time

    from .__main__ import load_keys
    from .config import KEYS_FILE

    load_keys(KEYS_FILE)
    key = os.environ.get("KR_API_KEY", "").strip()
    if not key:
        print("ai-keys.env에 KR_API_KEY가 없어요")
        return 1
    subprocess.run(["clip"], input=key.encode("ascii"), check=True)
    print(f"팀 키를 클립보드에 넣었어요. {int(keep_sec)}초 뒤에 비울게요", flush=True)
    end = time.time() + keep_sec
    while time.time() < end:
        time.sleep(1)
    if _clipboard().strip() == key:
        _clipboard(clear=True)
        print("클립보드를 비웠어요")
    else:
        print("그새 다른 걸 복사해서 클립보드는 그대로 뒀어요")
    return 0


def kr_key_new(keys_file: Path | None = None) -> int:
    """새 팀 키(KR_API_KEY)를 만들어 ai-keys.env에 적는다. 예전 키 줄은 새 키로 바뀐다. 키 값은 화면에 남기지 않는다.

    앱 APK에 들어간 팀 키가 새었을 때 쓴다. 바꾼 뒤에는 서버(Space Secret)와 앱 빌드에도 새 키를 넣어야 한다.
    """
    import re
    import secrets
    import string

    from .config import KEYS_FILE

    path = Path(keys_file or KEYS_FILE)
    raw = path.read_bytes() if path.exists() else b""
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in text else "\n"
    key = "kr-" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))
    line = f"KR_API_KEY={key}"
    # 주석이 아닌 KR_API_KEY 줄을 모두 바꾼다 (줄바꿈 문자는 그대로 둔다)
    pattern = re.compile(r"^[ \t]*KR_API_KEY[ \t]*=[^\r\n]*", re.M)
    text, n = pattern.subn(lambda _m: line, text)
    if not n:
        if text and not text.endswith("\n"):
            text += nl
        text += line + nl
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    print(f"새 팀 키를 {path.name}의 KR_API_KEY에 {'바꿔' if n else '새로'} 적었어요 (kr-****, {len(key)}자)")
    return 0


SPACE_SECRETS = ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "KR_API_KEY")


def hf_secrets(space: str, names: list[str] | None = None) -> int:
    """ai-keys.env의 키를 Space Secrets에 넣는다. 키 값은 화면에 남기지 않는다.

    names를 주면 그 키만 바꾸고 나머지 Secret과 Variable은 그대로 둔다.
    안 주면 키 3개를 모두 넣고 Variables에 MOCK_GENERATION=0을 넣는다.
    """
    import os

    chosen = tuple(names) if names else SPACE_SECRETS
    unknown = [k for k in chosen if k not in SPACE_SECRETS]
    if unknown:
        print("넣을 수 있는 키가 아니에요:", ", ".join(unknown), f"(가능: {', '.join(SPACE_SECRETS)})")
        return 2

    from huggingface_hub import HfApi

    from .__main__ import load_keys
    from .config import KEYS_FILE

    token = _hf_token()
    if not token:
        return 1
    load_keys(KEYS_FILE)
    missing = [k for k in chosen if not os.environ.get(k, "").strip()]
    if missing:
        print("ai-keys.env에 없는 키:", ", ".join(missing))
        return 1
    api = HfApi(token=token)
    space = _hf_space(api, space)
    for name in chosen:
        try:
            api.add_space_secret(space, name, os.environ[name].strip())
        except Exception as e:
            print(f"Secret {name}: 실패 {type(e).__name__} {_http_status(e)}")
            return 1
        print(f"Secret {name}: 넣음")
    if names:
        return 0
    try:
        api.add_space_variable(space, "MOCK_GENERATION", "0")
    except Exception as e:
        print(f"Variable MOCK_GENERATION: 실패 {type(e).__name__} {_http_status(e)}")
        return 1
    print("Variable MOCK_GENERATION=0: 넣음")
    return 0


def hf_deploy(space: str) -> int:
    """deploy/hf-space.tar(서버 코드, Dockerfile)를 Space에 올린다. 올리면 허깅페이스가 알아서 다시 빌드한다."""
    import tarfile
    import tempfile

    from huggingface_hub import HfApi

    token = _hf_token()
    if not token:
        return 1
    bundle = ROOT / "deploy" / "hf-space.tar"
    if not bundle.exists():
        print(f"올릴 묶음이 없어요: {bundle}")
        return 1
    api = HfApi(token=token)
    space = _hf_space(api, space)
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(bundle) as tf:
            try:
                tf.extractall(d, filter="data")
            except TypeError:
                tf.extractall(d)
        n = sum(1 for p in Path(d).rglob("*") if p.is_file())
        print(f"{space}에 파일 {n}개 올리는 중...")
        info = api.upload_folder(repo_id=space, repo_type="space", folder_path=d,
                                 commit_message="노래 생성 서버 올리기", delete_patterns=["*"])
    print("올림:", getattr(info, "commit_url", info))
    host = space.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    print(f"서버 주소: https://{host}.hf.space  (빌드가 끝나면 열려요)")
    return 0


def hf_pause(space: str) -> int:
    """Space를 멈춘다. 멈춘 동안은 하드웨어 요금이 안 나간다. 다시 켜려면 Space 설정의 Restart."""
    from huggingface_hub import HfApi

    token = _hf_token()
    if not token:
        return 1
    api = HfApi(token=token)
    space = _hf_space(api, space)
    try:
        rt = api.pause_space(space)
    except Exception as e:
        print(f"멈추지 못했어요: {type(e).__name__} {_http_status(e)}")
        return 1
    print(f"{space}: stage={rt.stage} (멈춤, 요금 안 나감)")
    return 0


def hf_restart(space: str) -> int:
    """멈춘 Space를 다시 켠다. 켜져 있는 동안 고른 하드웨어 요금이 나간다 (CPU upgrade는 시간당 $0.03)."""
    from huggingface_hub import HfApi

    token = _hf_token()
    if not token:
        return 1
    api = HfApi(token=token)
    space = _hf_space(api, space)
    try:
        rt = api.restart_space(space)
    except Exception as e:
        print(f"다시 켜지 못했어요: {type(e).__name__} {_http_status(e)}")
        return 1
    print(f"{space}: stage={rt.stage} hardware={rt.requested_hardware} (빌드가 끝나면 RUNNING이 돼요)")
    return 0


def hf_status(space: str) -> int:
    from huggingface_hub import HfApi

    token = _hf_token()
    if not token:
        return 1
    api = HfApi(token=token)
    space = _hf_space(api, space)
    rt = api.get_space_runtime(space)
    print(f"{space}: stage={rt.stage} hardware={rt.hardware} requested={rt.requested_hardware}")
    try:
        names = sorted(api.get_space_variables(space).keys())
        print("variables:", ", ".join(names) or "(없음)")
    except Exception as e:
        print("variables: 못 읽음", type(e).__name__)
    return 0


def hf_logs(space: str, kind: str = "build", seconds: float = 12.0) -> int:
    """빌드(build)나 실행(run) 기록 끝부분. 허깅페이스가 흘려주는 기록을 몇 초 받아서 보여 준다."""
    import json
    import socket
    import time
    import urllib.request

    from huggingface_hub import HfApi

    token = _hf_token()
    if not token:
        return 1
    space = _hf_space(HfApi(token=token), space)
    req = urllib.request.Request(f"https://huggingface.co/api/spaces/{space}/logs/{kind}",
                                 headers={"Authorization": f"Bearer {token}"})
    lines: list[str] = []
    deadline = time.time() + seconds
    try:
        with urllib.request.urlopen(req, timeout=seconds + 3) as r:
            while time.time() < deadline:
                raw = r.readline()
                if not raw:
                    break
                s = raw.decode("utf-8", "replace").strip()
                if s.startswith("data:"):
                    try:
                        lines.append(str(json.loads(s[5:]).get("data", "")).rstrip())
                    except ValueError:
                        lines.append(s[5:].strip())
    except (socket.timeout, TimeoutError):
        pass
    except Exception as e:
        print("기록을 못 읽었어요:", type(e).__name__, e)
    print("\n".join(x for x in lines[-80:] if x))
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "emu-record":
        if len(argv) < 2:
            print("사용: emu-record start|stop [이름] [초]")
            return 2
        return emu_record(argv[1], argv[2] if len(argv) > 2 else "demo", int(argv[3]) if len(argv) > 3 else 180)
    if argv and argv[0] == "app-prefs":
        if len(argv) < 3:
            print("사용: app-prefs get|put 이름 (예: app-prefs get family)")
            return 2
        return app_prefs(argv[1], argv[2])
    if argv and argv[0] == "screen-record":
        if len(argv) < 2:
            print("사용: screen-record start|stop|pull [이름] [초]")
            return 2
        return screen_record(argv[1], argv[2] if len(argv) > 2 else "demo", int(argv[3]) if len(argv) > 3 else 180)
    if argv and argv[0] == "make-test-audio":
        names = [a for a in argv[1:] if a != "fast"]
        return make_test_audio(Path(names[0]) if names else None, fast="fast" in argv[1:])
    if argv and argv[0] == "live-probe":
        if len(argv) < 2:
            print("사용: live-probe 주소 [파일] (예: live-probe https://awqdqwd-kidsrhythm-server.hf.space demo-family-fast.wav)")
            return 2
        return live_probe(argv[1], FILES / (argv[2] if len(argv) > 2 else "test-session.wav"))
    if argv and argv[0] == "check-keys":
        return check_keys()
    if argv and argv[0] == "tunnel-check":
        return tunnel_check(argv[1] if len(argv) > 1 else None)
    if argv and argv[0] == "kr-key-copy":
        return kr_key_copy(float(argv[1]) if len(argv) > 1 else 90.0)
    if argv and argv[0] == "kr-key-new":
        return kr_key_new()
    if argv and argv[0] == "hf-token-save":
        return hf_token_save(float(argv[1]) if len(argv) > 1 else 0.0)
    if argv and argv[0] in ("hf-deploy", "hf-status", "hf-logs", "hf-secrets", "hf-pause", "hf-restart"):
        if len(argv) < 2:
            print(f"Space 이름이 필요해요: python -m server.tools {argv[0]} kidsrhythm-server")
            return 2
        if argv[0] == "hf-deploy":
            return hf_deploy(argv[1])
        if argv[0] == "hf-status":
            return hf_status(argv[1])
        if argv[0] == "hf-secrets":
            return hf_secrets(argv[1], argv[2:] or None)
        if argv[0] == "hf-pause":
            return hf_pause(argv[1])
        if argv[0] == "hf-restart":
            return hf_restart(argv[1])
        return hf_logs(argv[1], argv[2] if len(argv) > 2 else "build")
    if argv and argv[0] == "words-probe":
        names = [a for a in argv[1:] if a != "fast"]
        return words_probe(FILES / (names[0] if names else "test-session.wav"), realtime="fast" not in argv[1:])
    if argv and argv[0] == "socket-stress":
        return socket_stress()
    if argv and argv[0] == "bench":
        return bench(argv[1:] or ["large-v3-turbo", "small"])
    if argv and argv[0] == "socket-test":
        if "fix" in argv[1:]:
            from .winsock_fix import install

            install()
        return socket_test(heavy="heavy" in argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
