"""노래 길이(절 수)와 여러 절 가사 검사 (서버 0.6.0). 모델 없이 돈다: python -m unittest discover -s tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai"))

from server.remote_song import (  # noqa: E402
    MAX_SONG_SEC, Lyrics, form_seconds, lyrics_system, plan_sections, song_form, song_verses, validate_lyrics)

OLD_FORM = [("Intro", "intro", 2, 0), ("Verse 1", "verse", 8, 4), ("Chorus", "chorus", 4, 2),
            ("Verse 2", "verse", 8, 4), ("Chorus", "chorus", 4, 2), ("Outro", "outro", 2, 1)]


class SongFormTest(unittest.TestCase):
    def test_verses_by_speech(self):
        self.assertEqual([song_verses(s) for s in (0, 14.9, 15, 29.9, 30, 90)], [2, 2, 3, 3, 4, 4])

    def test_never_over_three_minutes(self):
        for verses in (2, 3, 4):
            for bpm in (70, 80, 95, 110, 125, 140):
                with self.subTest(verses=verses, bpm=bpm):
                    self.assertLessEqual(form_seconds(song_form(verses, bpm), bpm), MAX_SONG_SEC)

    def test_short_form_is_the_old_one(self):
        self.assertEqual(song_form(2, 110), OLD_FORM)
        self.assertAlmostEqual(form_seconds(OLD_FORM, 110), 61.1, places=1)

    def test_longer_with_more_verses(self):
        lengths = [form_seconds(song_form(v, 110), 110) for v in (2, 3, 4)]
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[2], 150)
        long = song_form(4, 110)
        self.assertEqual(sum(1 for _, k, _, _ in long if k == "verse"), 4)
        self.assertIn("break", [k for _, k, _, _ in long])

    def test_slow_tempo_trims_but_keeps_verses(self):
        form = song_form(4, 80)
        self.assertEqual(sum(1 for _, k, _, _ in form if k == "verse"), 4)
        self.assertNotIn("break", [k for _, k, _, _ in form])
        self.assertLessEqual(form_seconds(form, 80), MAX_SONG_SEC)


class LyricsTest(unittest.TestCase):
    def data(self, verses: int) -> dict:
        d = {"title": "우리 바다 가자", "chorus": ["둥둥 바다 가자", "랄랄라 바다"], "outro": "바다 가자",
             "moodStyles": ["bouncy"]}
        for i in range(1, verses + 1):
            d[f"verse{i}"] = ["엄마 바다 가요", "강아지 같이 가요", "둥둥 바다", "랄랄라 가요"]
        return d

    def test_rules_follow_verse_count(self):
        self.assertIn("verse1과 verse2는 4줄씩", lyrics_system(2))
        self.assertNotIn("verse3", lyrics_system(2))
        four = lyrics_system(4)
        self.assertIn("verse1, verse2, verse3, verse4는 4줄씩", four)
        self.assertIn('"verse4": ["", "", "", ""]', four)

    def test_four_verses(self):
        lyr = validate_lyrics(self.data(4), ["바다"], "둥둥", "랄랄라", 10, verses=4)
        self.assertEqual(len(lyr.verses), 4)
        self.assertEqual(len(lyr.lines()), 4 * 4 + 4 * 2 + 1)
        with self.assertRaises(ValueError):
            validate_lyrics(self.data(3), ["바다"], "둥둥", "랄랄라", 10, verses=4)

    def test_plan_sings_chorus_twice_and_keeps_break_instrumental(self):
        lyr = Lyrics("t", [["가"] * 4 for _ in range(4)], ["후렴1", "후렴2"], "끝")
        sections = plan_sections(lyr, 110, song_form(4, 110))
        choruses = [s for s in sections if s.kind == "chorus"]
        self.assertTrue(all(s.lines == ["후렴1", "후렴2", "후렴1", "후렴2"] for s in choruses))
        self.assertEqual([s.lines for s in sections if s.kind == "break"], [[]])
        self.assertAlmostEqual(sum(s.dur_ms for s in sections) / 1000, form_seconds(song_form(4, 110), 110), delta=0.1)


if __name__ == "__main__":
    unittest.main()
