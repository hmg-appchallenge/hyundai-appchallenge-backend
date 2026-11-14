"""누가 말했는지 정하는 규칙 (서버 0.6.3).
다영님 확인: 다영님만 목소리를 등록하고 혼자 말했는데, 등록 안 한 정현·현중·세준 카드가 번갈아 켜졌다.
등록할 때와 녹음할 때 마이크가 다르면 같은 사람도 닮은 정도가 0.1~0.4로 떨어져서 생긴 일이다.
모델 없이 돈다: 닮은 정도(코사인)를 정해 둔 가짜 목소리 특징으로 본다.
python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))

from server.live import LiveSession, Member, Utterance  # noqa: E402
from server.speaker import StreamVad  # noqa: E402

DIM = 16


def axis(k: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[k] = 1.0
    return v


def voice(parts: dict[int, float]) -> np.ndarray:
    """축 번호: 값. 남은 크기는 맨 마지막 축(15)에 채워 길이 1로 만든다."""
    v = np.zeros(DIM, np.float32)
    for k, x in parts.items():
        v[k] = x
    rest = 1.0 - float(v @ v)
    v[DIM - 1] = math.sqrt(max(0.0, rest))
    return v


class FakeEngine:
    stream_asr = None

    def new_vad(self):
        return StreamVad(Path("/nonexistent"))


E1 = axis(0)  # 등록한 다영
E2 = axis(1)  # 등록한 다른 가족 (두 번째 시험용)


def session(*embs: np.ndarray | None) -> LiveSession:
    members = [Member(i + 1, "DRUM", e) for i, e in enumerate(embs)]
    return LiveSession("t", members, FakeEngine(), log=lambda *_: None)


def say(s: LiveSession, emb: np.ndarray, start: float, dur: float = 1.2) -> int | None:
    """닫힌 말소리 구간 하나 (녹음 중 판단 + 완성 때 다시 볼 구간으로 쌓기)."""
    mid = s.identify(emb, dur)
    s.utterances.append(Utterance(start, start + dur, np.zeros(1, np.float32), emb, mid))
    return mid


class SpeakerTest(unittest.TestCase):
    def test_enrolled_voice_on_other_mic_never_lights_unregistered_cards(self):
        s = session(E1, None, None, None)
        # 마이크가 달라 등록 목소리와 0.26~0.42만 닮은 다영님 말 (구간끼리도 서로 안 닮게)
        sims = [0.26, 0.31, 0.42, 0.28, 0.35, 0.30, 0.27, 0.39]
        got = [say(s, voice({0: x, 2 + i: math.sqrt(1 - x * x) * 0.99}), i * 2.0) for i, x in enumerate(sims)]
        self.assertEqual(got, [1] * len(sims))
        self.assertEqual(s.clusters, [])
        s.refine_speakers()
        self.assertEqual([u.member_id for u in s.utterances], [1] * len(sims))

    def test_clearly_different_long_voice_becomes_first_unregistered(self):
        s = session(E1, None, None, None)
        other = voice({0: 0.05, 5: 0.9})
        self.assertEqual(say(s, other, 0.0, 1.5), 2)
        again = voice({0: 0.02, 5: 0.85, 6: 0.3})  # 같은 사람 (앞 구간과 0.8 넘게 닮음)
        self.assertEqual(say(s, again, 2.0, 1.1), 2)
        s.refine_speakers()
        self.assertEqual([u.member_id for u in s.utterances], [2, 2])

    def test_partial_chunk_does_not_create_a_new_person(self):
        s = session(E1, None, None, None)
        self.assertIsNone(s.identify(voice({5: 0.95}), 1.6, final=False))
        self.assertEqual(s.clusters, [])

    def test_short_different_voice_does_not_create_a_new_person(self):
        s = session(E1, None, None, None)
        self.assertIsNone(say(s, voice({5: 0.95}), 0.0, 0.7))
        self.assertEqual(s.clusters, [])

    def test_learns_this_recordings_voice(self):
        s = session(E1, None, None, None)
        a = voice({0: 0.40, 2: math.sqrt(0.84) * 0.999})
        self.assertEqual(say(s, a, 0.0, 1.2), 1)  # 등록 목소리와 0.40: 이번 녹음 목소리로 익힌다
        self.assertIn(1, s.adapted)
        # 등록 목소리와는 0.10만 닮았지만 방금 익힌 목소리와 0.75 닮은 구간 -> 다영님
        b = np.zeros(DIM, np.float32)
        b[0], b[2] = 0.10, (0.75 - 0.40 * 0.10) / float(a[2])
        b[3] = math.sqrt(1 - float(b @ b))
        self.assertAlmostEqual(float(b @ a), 0.75, delta=0.01)
        self.assertEqual(say(s, b, 2.0, 1.0), 1)

    def test_other_person_on_same_mic_is_not_absorbed(self):
        s = session(E1, None, None, None)
        a = voice({0: 0.40, 2: math.sqrt(0.84) * 0.999})
        say(s, a, 0.0, 1.2)
        # 같은 마이크라 익힌 목소리와 0.42 닮지만 등록 목소리와는 0.15인 엄마 -> 등록 안 한 가족
        m = np.zeros(DIM, np.float32)
        m[0], m[2] = 0.15, (0.42 - 0.40 * 0.15) / float(a[2])
        m[4] = math.sqrt(1 - float(m @ m))
        self.assertEqual(say(s, m, 2.0, 1.4), 2)
        s.refine_speakers()
        self.assertEqual([u.member_id for u in s.utterances], [1, 2])

    def test_long_segment_is_not_used_to_learn(self):
        # 6초 넘게 이어진 구간은 여러 사람이 섞이기 쉬워서 이번 녹음 목소리로 익히지 않는다
        s = session(E1, None, None, None)
        self.assertEqual(say(s, voice({0: 0.40, 2: 0.9}), 0.0, 6.5), 1)
        self.assertEqual(s.adapted, {})

    def test_refine_leaves_tiny_different_blip_unknown(self):
        s = session(E1, None, None, None)
        say(s, voice({0: 0.33, 2: 0.9}), 0.0, 1.5)
        say(s, voice({0: 0.30, 3: 0.9}), 2.0, 1.2)
        s.utterances.append(Utterance(4.0, 4.5, np.zeros(1, np.float32), voice({7: 0.95}), None))  # 0.5초 기침
        s.refine_speakers()
        self.assertEqual([u.member_id for u in s.utterances], [1, 1, None])

    def test_nobody_enrolled_assigns_in_speaking_order(self):
        s = session(None, None, None)
        x, y = voice({3: 0.95}), voice({8: 0.95})
        self.assertEqual(say(s, x, 0.0, 1.0), 1)
        self.assertEqual(say(s, y, 2.0, 1.0), 2)
        self.assertEqual(say(s, x, 4.0, 1.0), 1)
        s.refine_speakers()
        self.assertEqual([u.member_id for u in s.utterances], [1, 2, 1])

    def test_everyone_enrolled_unknown_voice_is_flagged(self):
        s = session(E1, E2)
        self.assertIsNone(s.identify(voice({0: 0.10, 1: 0.05, 9: 0.9}), 1.2))
        self.assertTrue(s.unknown_flag)
        self.assertEqual(s.identify(voice({1: 0.30, 9: 0.9}), 1.2), 2)


if __name__ == "__main__":
    unittest.main()
