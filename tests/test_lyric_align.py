"""가사 줄 시간이 ElevenLabs가 실제로 부른 단어 시간을 따라가는지 (서버 0.6.2).
박자표로만 나누면 노래가 앞당기거나 늦게 부를 때 앱 가사 표시가 빨랐다 느렸다 한다.
python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))

from server.remote_song import LEAD_MS, Section, _timings, align_lines  # noqa: E402

BPM = 110
BAR = 240.0 / BPM  # 초


def sung(lines_with_start: list[tuple[str, float]], gap: float = 0.32) -> list[tuple[float, float, str]]:
    """줄마다 시작 시간을 주면 띄어쓰기 단위 단어 시간을 만든다 (ElevenLabs words_timestamps 흉내)."""
    out = []
    for text, t in lines_with_start:
        for w in text.split():
            out.append((t, t + gap, w))
            t += gap + 0.05
    return out


def sections(verse: list[str], chorus: list[str]) -> list[Section]:
    intro = Section("Intro", "intro", 2, [], 0, int(2 * BAR * 1000))
    v = Section("Verse 1", "verse", 2 * len(verse), verse, intro.dur_ms, int(2 * len(verse) * BAR * 1000))
    c = Section("Chorus", "chorus", 2 * len(chorus), chorus, v.start_ms + v.dur_ms, int(2 * len(chorus) * BAR * 1000))
    return [intro, v, c]


VERSE = ["엄마 나 사과 먹고 싶어", "사과는 빨갛고 맛있지"]
CHORUS = ["사과 사과 둥둥둥둥", "랄랄라 신난다", "사과 사과 둥둥둥둥", "랄랄라 신난다"]


class AlignTest(unittest.TestCase):
    def grid(self, secs: list[Section], first_beat: float) -> list[int]:
        return [it["startMs"] for it in _timings(secs, first_beat, [])]

    def test_follows_early_and_late_singing(self):
        secs = sections(VERSE, CHORUS)
        grid = self.grid(secs, 0.3)
        # 앞당겨 부른 줄(-0.9초)과 늦게 부른 줄(+0.7초)이 섞여 있다
        shifts = [-0.9, 0.7, 0.0, -0.5, 0.4, -0.2]
        texts = VERSE + CHORUS
        words = sung([(t, g / 1000 + d) for t, g, d in zip(texts, grid, shifts)])
        got = _timings(secs, 0.3, words)
        for it, g, d in zip(got, grid, shifts):
            self.assertAlmostEqual(it["startMs"], g + d * 1000 - LEAD_MS, delta=2)
        starts = [it["startMs"] for it in got]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(got, got[1:]):
            self.assertEqual(a["endMs"], b["startMs"])

    def test_repeated_chorus_lines_get_their_own_time(self):
        secs = sections(VERSE, CHORUS)
        grid = self.grid(secs, 0.0)
        texts = VERSE + CHORUS
        words = sung([(t, g / 1000 + 0.3) for t, g in zip(texts, grid)])
        got = _timings(secs, 0.0, words)
        self.assertNotEqual(got[2]["startMs"], got[4]["startMs"])
        self.assertAlmostEqual(got[4]["startMs"], grid[4] + 300 - LEAD_MS, delta=2)

    def test_skipped_line_keeps_order(self):
        secs = sections(VERSE, CHORUS)
        grid = self.grid(secs, 0.0)
        texts = VERSE + CHORUS
        # 노래가 3번째 줄을 건너뛰었다
        words = sung([(t, g / 1000 + 0.5) for i, (t, g) in enumerate(zip(texts, grid)) if i != 2])
        got = _timings(secs, 0.0, words)
        starts = [it["startMs"] for it in got]
        self.assertEqual(starts, sorted(starts))
        self.assertAlmostEqual(got[3]["startMs"], grid[3] + 500 - LEAD_MS, delta=2)

    def test_ad_libs_between_lines_are_ignored(self):
        lines = ["우리 같이 노래하자", "바다 바다 철썩철썩"]
        words = [(1.0, 1.3, "우리"), (1.35, 1.6, "같이"), (1.65, 2.2, "노래하자"),
                 (2.6, 2.9, "예"), (3.0, 3.2, "와아"),
                 (4.1, 4.4, "바다"), (4.45, 4.7, "바다"), (4.8, 5.4, "철썩철썩")]
        spans = align_lines(lines, words)
        self.assertAlmostEqual(spans[0][0], 1.0)
        self.assertAlmostEqual(spans[1][0], 4.1)
        self.assertAlmostEqual(spans[1][1], 5.4)

    def test_no_words_falls_back_to_bar_grid(self):
        secs = sections(VERSE, CHORUS)
        got = _timings(secs, 0.25, [])
        self.assertEqual(got[0]["startMs"], 250 + secs[1].start_ms)
        self.assertAlmostEqual(got[1]["startMs"] - got[0]["startMs"], 2 * BAR * 1000, delta=1)

    def test_one_matching_syllable_is_not_trusted(self):
        spans = align_lines(["엄마 나 사과 먹고 싶어"], [(3.0, 3.2, "나")])
        self.assertIsNone(spans[0])


if __name__ == "__main__":
    unittest.main()
