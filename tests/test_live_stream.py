"""실시간 받아쓰기로 말한 그 조각 응답에 단어가 오는지 (서버 0.6.1).
모델 없이 돈다: python -m unittest discover -s tests (Kiwi가 깔려 있으면 거르는 규칙도 본다)
"""
from __future__ import annotations

import re
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))

from server.live import LiveSession, Member  # noqa: E402
from server.speaker import StreamVad  # noqa: E402

SR = 16000
ME = np.array([1.0, 0.0], np.float32)
VOCAB = re.compile("강아지|엄마|바다|둥둥|강아")  # 가짜 형태소 분석: 긴 말부터 찾는다


def voice(sec: float = 2.0) -> np.ndarray:
    """말소리 흉내: 0.6초 소리 + 0.2초 쉼. 쉼이 짧아 구간이 끊기지 않는다."""
    t = np.arange(int(sec * SR)) / SR
    on = (t % 0.8) < 0.6
    return (0.3 * np.sin(2 * np.pi * 300 * t) * on).astype(np.float32)


def quiet(sec: float = 2.0) -> np.ndarray:
    return np.zeros(int(sec * SR), np.float32)


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


class FakeEncoder:
    name, dim = "fake", 2

    def embed(self, x):
        return ME.copy() if len(x) >= 1600 else None


class FakeStreamAsr:
    """조각마다 정해 둔 (받아쓴 글, 마디 끝) 을 차례로 돌려준다."""
    ready, name = True, "fake"

    def __init__(self, results):
        self.results = list(results)
        self.fed = 0

    def new_stream(self):
        return object()

    def feed(self, stream, x):
        self.fed += 1
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeEngine:
    def __init__(self, results, final_words=()):
        self.encoder = FakeEncoder()
        self.live_executor = ThreadPoolExecutor(1)
        self.stream_asr = FakeStreamAsr(results)
        self.final_words = list(final_words)

    def new_vad(self):
        return StreamVad(Path("/nonexistent"))  # 소리 크기로 구간을 찾는다

    def stream_notable(self, text, member_id):
        return [{"text": m.group(), "memberId": member_id, "kind": "word"} for m in VOCAB.finditer(text)]

    def transcribe_utterance(self, audio, offset):
        from songmaker.text import Word

        return [Word(t, offset + 0.5, offset + 0.9, 0.9) for t in self.final_words]

    def notable_words(self, words, member_id):
        return [{"text": w.text, "memberId": member_id, "kind": "word"} for w in words]


class StreamWordsTest(unittest.TestCase):
    def session(self, results, final_words=()):
        self.engine = FakeEngine(results, final_words)
        self.addCleanup(self.engine.live_executor.shutdown, True)
        return LiveSession("t", [Member(1, "DRUM", ME.copy())], self.engine, log=lambda *_: None)

    def test_words_arrive_in_the_same_chunk(self):
        s = self.session([("엄마우리어디가? 강아", False),
                          ("엄마우리어디가? 강아지도같이가요. 바다", False),
                          ("엄마우리어디가? 강아지도같이가요. 바다", False)])
        r0 = s.add_chunk(0, pcm(voice()))
        self.assertEqual(r0["newWords"], [{"text": "엄마", "memberId": 1, "kind": "word"}])
        # 글 끝의 '강아'는 더 이어질 수 있어서 미뤘다가, 다음 조각에서 '강아지'로 온다
        r1 = s.add_chunk(1, pcm(voice()))
        self.assertEqual([w["text"] for w in r1["newWords"]], ["강아지"])
        # 글이 그대로면(더 말하지 않음) 끝 단어도 보낸다. 누가 말했는지 모르면 memberId는 None
        r2 = s.add_chunk(2, pcm(quiet()))
        self.assertEqual(r2["newWords"], [{"text": "바다", "memberId": None, "kind": "word"}])

    def test_end_of_phrase_sends_last_word_and_starts_over(self):
        s = self.session([("둥둥", True), ("엄마", False), ("엄마둥둥", True)])
        self.assertEqual([w["text"] for w in s.add_chunk(0, pcm(voice()))["newWords"]], ["둥둥"])
        self.assertEqual(s.stream_text, "")
        self.assertEqual(s.add_chunk(1, pcm(voice()))["newWords"], [])  # 새 마디 끝의 '엄마'는 기다린다
        self.assertEqual([w["text"] for w in s.add_chunk(2, pcm(voice()))["newWords"]], ["엄마"])  # 둥둥은 이미 보냄

    def test_whisper_does_not_resend_same_word(self):
        s = self.session([("엄마바다", True), ("", False), ("", False)], final_words=["엄마", "기차"])
        first = s.add_chunk(0, pcm(np.concatenate([voice(1.0), quiet(1.0)])))["newWords"]
        self.engine.live_executor.submit(lambda: None).result(timeout=10)
        later = s.add_chunk(1, pcm(quiet()))["newWords"] + s.add_chunk(2, pcm(quiet()))["newWords"]
        self.assertEqual([w["text"] for w in first], ["엄마", "바다"])
        self.assertEqual([w["text"] for w in later], ["기차"])  # 구간을 다시 들은 결과에서 새 말만

    def test_recognizer_error_keeps_session_going(self):
        s = self.session([RuntimeError("boom")])
        r = s.add_chunk(0, pcm(voice()))
        self.assertEqual(r["newWords"], [])
        self.assertIsNone(s.stream)
        s.add_chunk(1, pcm(voice()))
        self.assertEqual(self.engine.stream_asr.fed, 1)

    def test_without_model_nothing_changes(self):
        engine = FakeEngine([])
        self.addCleanup(engine.live_executor.shutdown, True)
        engine.stream_asr.ready = False
        s = LiveSession("t", [Member(1, "DRUM", ME.copy())], engine, log=lambda *_: None)
        self.assertIsNone(s.stream)
        self.assertEqual(s.add_chunk(0, pcm(voice()))["newWords"], [])


try:
    import kiwipiepy  # noqa: F401

    HAS_KIWI = True
except ImportError:
    HAS_KIWI = False


@unittest.skipUnless(HAS_KIWI, "Kiwi가 없으면 건너뛴다")
class StrictFilterTest(unittest.TestCase):
    """실시간 받아쓰기가 실제로 낸 글(띄어쓰기 없음, 잘못 들은 말 섞임)에서 고르는 규칙."""

    @classmethod
    def setUpClass(cls):
        from kiwipiepy import Kiwi

        from server.engines import RealEngine

        cls.engine = RealEngine.__new__(RealEngine)  # 모델은 안 불러오고 Kiwi만 쓴다
        cls.engine.kiwi = Kiwi()

    def words(self, text):
        return [w["text"] for w in self.engine.stream_notable(text, None)]

    def test_keeps_real_words_in_spoken_order(self):
        got = self.words("엄마우리어디가? 바다보러가지. 와바다릉릉출발. 둥둥알랄라바다에서조개축장. 강아지도같이가요. 멍멍.")
        for w in ("엄마", "바다", "강아지", "조개", "둥둥", "멍멍"):
            self.assertIn(w, got)
        self.assertLess(got.index("엄마"), got.index("강아지"))

    def test_drops_long_glued_unknown_words(self):
        got = self.words("와큰억이다공용이다크앙거울보고머리빗자")
        self.assertNotIn("와큰억이다공용이다크앙거울", got)
        self.assertTrue(all(len(w) <= 4 for w in got), got)

    def test_drops_words_not_in_dictionary(self):
        got = self.words("오줌마려화장실! 랄랄라장밖에무지게떴다.")
        self.assertIn("화장실", got)
        self.assertNotIn("오줌마려", got)
        self.assertNotIn("랄랄라장", got)

    def test_sound_words_must_be_made_of_sound_syllables(self):
        got = self.words("와바다릉릉출발. 무지개예쁘다, 등등. 부릉부릉 둥둥둥")
        self.assertNotIn("릉릉", got)
        self.assertNotIn("등등", got)
        self.assertIn("부릉부릉", got)
        self.assertIn("둥둥둥", got)


if __name__ == "__main__":
    unittest.main()
