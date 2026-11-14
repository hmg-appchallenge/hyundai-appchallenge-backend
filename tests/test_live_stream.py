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

    def test_misheard_real_words_wait_for_whisper(self):
        # 실시간 받아쓰기가 실제로 잘못 들은 말: 토끼 -> 토기, 공룡 -> 공용. 사전에는 있는 말이라 '아는 말'로 거른다
        got = self.words("토기가깡충깡충뛰어요. 와큰억이다공용이다. 자루")
        self.assertEqual(got, ["깡충깡충"])

    def test_animal_sounds(self):
        got = self.words("고양이가야옹야옹울어요. 비행기가슝슝날아가요")
        self.assertEqual(got, ["고양이", "야옹야옹", "비행기", "슝슝"])


class OnomatopoeiaTest(unittest.TestCase):
    """현중님 시험(11/14)에서 '고양이가 야옹야옹 울어요'의 야옹야옹이 빠졌다. 인식은 됐는데 의성어로 못 알아봤다."""

    def test_repeated_two_syllable_sounds(self):
        from songmaker.text import is_onomatopoeia

        for w in ("야옹야옹", "부엉부엉", "깡충깡충", "데굴데굴", "히힝", "슝슝", "멍멍"):
            self.assertTrue(is_onomatopoeia(w), w)

    @unittest.skipUnless(HAS_KIWI, "Kiwi가 없으면 건너뛴다")
    def test_whisper_sentence_keeps_meow_but_not_noun_repeats(self):
        from kiwipiepy import Kiwi

        from songmaker.text import Word, analyze, is_onomatopoeia

        kiwi = Kiwi()
        an = analyze([Word(t, 0, 0, 0.99) for t in "고양이가 야옹야옹 울어요.".split()], "", kiwi)
        self.assertEqual([n.text for n in an.nouns], ["고양이"])
        self.assertEqual([o.text for o in an.onos], ["야옹야옹"])
        self.assertFalse(is_onomatopoeia("엄마엄마", kiwi))


class PaddingTest(unittest.TestCase):
    """Whisper에는 말소리 구간 앞뒤를 조금 붙여 들려준다 (첫소리가 잘리면 토끼가 안 들리거나 아빠랑이 파랑이 된다)."""

    def test_utterance_audio_has_pre_and_post_roll_without_overlap(self):
        engine = FakeEngine([])
        self.addCleanup(engine.live_executor.shutdown, True)
        engine.stream_asr.ready = False
        s = LiveSession("t", [Member(1, "DRUM", ME.copy())], engine, log=lambda *_: None)
        tone = np.sin(2 * np.pi * 300 * np.arange(int(0.4 * SR)) / SR).astype(np.float32) * 0.3
        x = np.concatenate([quiet(1.0), tone, quiet(0.6), tone, quiet(1.6)])  # 1.0~1.4초, 2.0~2.4초에 말
        s.add_chunk(0, pcm(x))
        a, b = s.utterances
        self.assertAlmostEqual(a.start, 1.0, places=2)
        self.assertAlmostEqual(a.audio_start, 0.7, places=2)  # 앞 0.3초
        self.assertAlmostEqual(len(a.audio) / SR, 0.4 + 0.3 + 0.15, places=2)  # 뒤 0.15초
        self.assertAlmostEqual(b.audio_start, 1.7, places=2)  # 앞 구간이 넘긴 1.55초보다 뒤라 0.3초를 다 붙인다
        self.assertAlmostEqual(a.duration, 0.4, places=2)  # 말한 시간(노래 길이 계산)은 그대로
        engine.live_executor.submit(lambda: None).result(timeout=10)

    def test_close_utterances_do_not_share_audio(self):
        engine = FakeEngine([])
        self.addCleanup(engine.live_executor.shutdown, True)
        engine.stream_asr.ready = False
        s = LiveSession("t", [Member(1, "DRUM", ME.copy())], engine, log=lambda *_: None)
        tone = np.sin(2 * np.pi * 300 * np.arange(int(0.4 * SR)) / SR).astype(np.float32) * 0.3
        x = np.concatenate([quiet(1.0), tone, quiet(0.35), tone, quiet(1.0)])  # 쉼이 0.35초뿐
        s.add_chunk(0, pcm(x))
        a, b = s.utterances
        self.assertLessEqual(a.audio_start + len(a.audio) / SR, b.audio_start + 1e-6)
        engine.live_executor.submit(lambda: None).result(timeout=10)


if __name__ == "__main__":
    unittest.main()
