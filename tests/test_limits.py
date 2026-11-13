"""노래 만들기 횟수 제한과 제목 길이 맞추기 시험. 모델 없이 돈다: python -m unittest discover -s tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))

from server.limits import SongLimiter, client_ip  # noqa: E402
from songmaker.text import fit_title  # noqa: E402


class Clock:
    def __init__(self, t: float = 1_794_500_000.0):  # 2026-11-13 낮 (한국 시간)
        self.t = t

    def __call__(self) -> float:
        return self.t


class SongLimiterTest(unittest.TestCase):
    def test_한_IP는_한_시간에_정한_수까지만_만든다(self):
        clock = Clock()
        lim = SongLimiter(per_day=100, per_hour_per_ip=2, clock=clock)
        self.assertIsNone(lim.try_take("1.1.1.1"))
        self.assertIsNone(lim.try_take("1.1.1.1"))
        self.assertIn("한 시간에 2곡", lim.try_take("1.1.1.1"))
        self.assertIsNone(lim.try_take("2.2.2.2"))  # 다른 곳은 따로 센다
        clock.t += 3600
        self.assertIsNone(lim.try_take("1.1.1.1"))  # 한 시간이 지나면 다시 된다

    def test_하루_전체_한도를_넘으면_IP가_달라도_막는다(self):
        clock = Clock()
        lim = SongLimiter(per_day=3, per_hour_per_ip=10, clock=clock)
        for ip in ("a", "b", "c"):
            self.assertIsNone(lim.try_take(ip))
        self.assertIn("오늘 만들 수 있는 노래(3곡)", lim.try_take("d"))
        self.assertEqual(lim.status()["songsToday"], 3)

    def test_한국_자정이_지나면_하루_횟수가_다시_시작된다(self):
        clock = Clock()
        lim = SongLimiter(per_day=1, per_hour_per_ip=0, clock=clock)
        self.assertIsNone(lim.try_take("a"))
        self.assertIsNotNone(lim.try_take("a"))
        clock.t += 24 * 3600
        self.assertIsNone(lim.try_take("a"))

    def test_0이면_제한하지_않는다(self):
        lim = SongLimiter(per_day=0, per_hour_per_ip=0, clock=Clock())
        for _ in range(500):
            self.assertIsNone(lim.try_take("a"))

    def test_프록시_머리글에서_원래_IP를_고른다(self):
        self.assertEqual(client_ip({"cf-connecting-ip": "9.9.9.9", "x-forwarded-for": "1.1.1.1"}, "127.0.0.1"), "9.9.9.9")
        self.assertEqual(client_ip({"x-forwarded-for": "5.5.5.5, 10.0.0.1"}, "127.0.0.1"), "5.5.5.5")
        self.assertEqual(client_ip({}, "192.168.0.7"), "192.168.0.7")


class FitTitleTest(unittest.TestCase):
    def test_12자_이내는_그대로(self):
        self.assertEqual(fit_title("우리 바다 가자"), "우리 바다 가자")
        self.assertEqual(fit_title("  둥둥   자동차 "), "둥둥 자동차")

    def test_넘치면_띄어쓰기에서_끊는다(self):
        t = fit_title("우리 가족 신나는 드라이브 노래")
        self.assertEqual(t, "우리 가족 신나는")
        self.assertLessEqual(len(t), 12)

    def test_띄어쓰기가_없으면_12자에서_자른다(self):
        self.assertEqual(fit_title("아이스크림케이크노래자랑대회"), "아이스크림케이크노래자랑")

    def test_날짜_제목도_12자_안이다(self):
        self.assertLessEqual(len("12월 31일 드라이브"), 12)


if __name__ == "__main__":
    unittest.main()
