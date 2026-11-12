"""노래를 실제로 만드는 쪽.

RealEngine: 음성 인식(faster-whisper) + 형태소 분석(Kiwi) + TTS(Supertonic) + 화자 모델로 진짜 노래를 만든다.
MockEngine: 모델 없이 정해진 단어·화자를 흉내 낸다. 앱 연동을 먼저 시험할 때 쓴다.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import AI_DIR, MODELS_DIR, SHERPA_DIR, SR_IN
from .live import LiveSession, Member, Utterance
from .speaker import SpeakerEncoder, StreamVad

if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from songmaker.dsp import SR, resample  # noqa: E402
from songmaker.session_song import MemberData, make_session_song  # noqa: E402
from songmaker.text import Analysis, Noun, Ono, Word, analyze  # noqa: E402


class NotReady(Exception):
    pass


def member_data(utterances: list[Utterance], members: list[Member], analyze_fn) -> list[MemberData]:
    by_id: dict[int, list[Utterance]] = {m.member_id: [] for m in members}
    unknown: list[Utterance] = []
    for u in utterances:
        (by_id[u.member_id] if u.member_id in by_id else unknown).append(u)
    out = []
    for m in members:
        us = by_id[m.member_id]
        words = [w for u in us for w in u.words]
        out.append(MemberData(m.member_id, m.instrument, words, analyze_fn(words, str(m.member_id)),
                              speech_sec=float(sum(u.duration for u in us))))
    if unknown:
        words = [w for u in unknown for w in u.words]
        out.append(MemberData(0, "", words, analyze_fn(words, ""), speech_sec=float(sum(u.duration for u in unknown))))
    return out


class BaseEngine:
    name = "base"
    session_class = LiveSession

    def __init__(self, log=print):
        self.log = log
        self.encoder = SpeakerEncoder(SHERPA_DIR, log=log)
        self.live_executor = ThreadPoolExecutor(1, thread_name_prefix="live")
        self.song_executor = ThreadPoolExecutor(1, thread_name_prefix="song")
        self.ready = threading.Event()
        self.load_error: str | None = None

    def new_vad(self) -> StreamVad:
        return StreamVad(SHERPA_DIR)

    def wait_ready(self, timeout: float) -> None:
        if not self.ready.wait(timeout):
            raise NotReady(self.load_error or "모델을 아직 불러오는 중")
        if self.load_error:
            raise NotReady(self.load_error)

    def status(self) -> dict:
        return {"mode": self.name, "ready": self.ready.is_set() and not self.load_error,
                "speakerModel": self.encoder.name, "vad": self.new_vad().kind, "error": self.load_error}


class RealEngine(BaseEngine):
    name = "real"

    def __init__(self, log=print, stt_model: str = "large-v3-turbo", voice: str = "F1", pitch: float = 3.0,
                 tts_steps: int = 10):
        super().__init__(log)
        self.stt_model, self.voice_name, self.pitch, self.tts_steps = stt_model, voice, pitch, tts_steps
        self.stt = self.kiwi = self.voice = None
        self._stt_lock = threading.Lock()
        threading.Thread(target=self._load, name="load-models", daemon=True).start()

    def _load(self) -> None:
        t0 = time.time()
        try:
            from kiwipiepy import Kiwi

            self.kiwi = Kiwi()
            from songmaker.stt import Transcriber

            self.log(f"음성 인식 모델 불러오는 중 ({self.stt_model}) ...")
            self.stt = Transcriber(self.stt_model, MODELS_DIR / "whisper")
            from songmaker.tts import SupertonicTTS, Voice

            self.log(f"TTS 불러오는 중 (Supertonic-3, {self.voice_name}) ...")
            tts = SupertonicTTS(self.voice_name, MODELS_DIR, steps=self.tts_steps)
            self.voice = Voice(tts, pitch=self.pitch)
            self.log(f"모델 준비 끝 ({time.time() - t0:.1f}초). 화자 모델: {self.encoder.name}")
        except Exception as e:
            self.load_error = f"모델을 불러오지 못했어: {type(e).__name__}: {e}"
            self.log("[멈춤] " + self.load_error)
        finally:
            self.ready.set()

    def analyze_words(self, words: list[Word], label: str) -> Analysis:
        return analyze(words, label, self.kiwi)

    def transcribe_utterance(self, audio16: np.ndarray, offset: float) -> list[Word]:
        self.wait_ready(120)
        pad = 0.15
        x = np.concatenate([np.zeros(int(pad * SR_IN), np.float32), audio16, np.zeros(int(pad * SR_IN), np.float32)])
        with self._stt_lock:
            words = self.stt.transcribe(x)
        base = offset - pad
        return [Word(w.text, max(0.0, w.start + base), max(0.0, w.end + base), w.prob) for w in words]

    def notable_words(self, words: list[Word], member_id: int | None) -> list[dict]:
        an = self.analyze_words(words, str(member_id) if member_id is not None else "")
        return ([{"text": n.text, "memberId": member_id, "kind": "word"} for n in an.nouns]
                + [{"text": o.text, "memberId": member_id, "kind": "onomatopoeia"} for o in an.onos])

    def make_song(self, audio16: np.ndarray, utterances: list[Utterance], members: list[Member],
                  drive: dict | None, beat: dict | None, progress, log) -> tuple[np.ndarray, dict]:
        self.wait_ready(120)
        data = member_data(utterances, members, self.analyze_words)
        audio44 = resample(audio16, SR_IN, SR)
        players = [(m.member_id, m.instrument) for m in members]
        from .remote_song import make_remote_song, remote_enabled

        if remote_enabled():
            try:
                return make_remote_song(audio44, data, players, drive, beat, utterances, progress, log, kiwi=self.kiwi)
            except Exception as e:
                if type(e).__name__ in ("Cancelled", "JobTimeout"):
                    raise
                log(f"[외부 생성 실패 -> PC 엔진으로 만들어요] {type(e).__name__}: {str(e)[:300]}")
        return make_session_song(audio44, data, players, drive, beat, self.voice, progress, log)

    def status(self) -> dict:
        from .remote_song import remote_enabled

        return {**super().status(), "generator": "claude+elevenlabs" if remote_enabled() else "local"}


# ---------- 흉내 내는 쪽 ----------
MOCK_WORDS = [("엄마", "word"), ("둥둥", "onomatopoeia"), ("랄랄라", "onomatopoeia"), ("사과", "word"), ("강아지", "word")]
MOCK_SPEAKERS = [(0, 1), (1,), (2,), (0, 2), ()]  # 2초마다 바뀌는 '지금 소리 낸 사람' (가족 순서)


class MockLiveSession(LiveSession):
    """들어온 소리와 상관없이 정해진 순서로 화자와 단어를 돌려준다."""

    def add_chunk(self, seq: int, data: bytes) -> dict:
        with self.lock:
            self.last_activity = time.time()
            fresh = seq >= self.next_seq
            if fresh:
                self.next_seq = seq + 1
                x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
                self.chunks.append(x)
                self.samples += len(x)
            ids = [m.member_id for m in self.members]
            speaking = [ids[i] for i in MOCK_SPEAKERS[seq % len(MOCK_SPEAKERS)] if i < len(ids)] if fresh else []
            words = []
            if fresh and seq % 2 == 1 and seq // 2 < len(MOCK_WORDS):
                text, kind = MOCK_WORDS[seq // 2]
                words.append({"text": text, "memberId": ids[(seq // 2) % len(ids)] if ids else None, "kind": kind})
            return {"seq": seq, "receivedSec": round(self.received_sec, 2), "speaking": speaking,
                    "unregisteredVoice": False, "newWords": words}

    def finalize(self, timeout: float = 90.0) -> list[Utterance]:
        with self.lock:
            self.finished = True
        return []


class MockEngine(BaseEngine):
    name = "mock"
    session_class = MockLiveSession

    def __init__(self, log=print, step_delay: float = 1.0):
        super().__init__(log)
        from songmaker.tts import ToneTTS, Voice

        self.voice = Voice(ToneTTS(), pitch=0.0)
        self.step_delay = step_delay
        self.ready.set()

    def analyze_words(self, words: list[Word], label: str) -> Analysis:
        return Analysis(label, " ".join(w.text for w in words), [], [], [])

    def transcribe_utterance(self, audio16: np.ndarray, offset: float) -> list[Word]:
        return []

    def notable_words(self, words: list[Word], member_id: int | None) -> list[dict]:
        return []

    def make_song(self, audio16: np.ndarray, utterances: list[Utterance], members: list[Member],
                  drive: dict | None, beat: dict | None, progress, log) -> tuple[np.ndarray, dict]:
        ids = [m.member_id for m in members] or [1]

        def pick(i: int) -> int:
            return ids[i % len(ids)]

        progress("FIND_WORDS", 0.2)
        time.sleep(self.step_delay)
        canned: dict[int, tuple[list, list]] = {}
        for i, (word, kind) in enumerate([("둥둥", "kick"), ("사과", "noun"), ("랄랄라", "melody")]):
            mid = pick(i)
            nouns_i, onos_i = canned.setdefault(mid, ([], []))
            if kind == "noun":
                nouns_i.append(Noun(word, "object", str(mid), 1.5, 2.0))
            else:
                onos_i.append(Ono(word, kind, str(mid), 0.5 + i, 1.0 + i))
        data = []
        for m in members:
            nouns, onos = canned.get(m.member_id, ([], []))
            data.append(MemberData(m.member_id, m.instrument, [], Analysis(str(m.member_id), "", nouns, onos, []),
                                   speech_sec=2.0 if (nouns or onos) else 0.0))

        def slow_progress(step: str, p: float) -> None:
            progress(step, p)
            time.sleep(self.step_delay)

        silence = np.zeros(SR, np.float32)
        return make_session_song(silence, data, [(m.member_id, m.instrument) for m in members], drive, beat,
                                 self.voice, slow_progress, log)
