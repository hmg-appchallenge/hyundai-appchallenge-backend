"""kr-key-new가 ai-keys.env의 다른 줄은 그대로 두고 팀 키 줄만 바꾸는지 본다."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from server.tools import kr_key_new
except ImportError as e:  # numpy 없는 환경
    kr_key_new = None
    SKIP = str(e)


@unittest.skipIf(kr_key_new is None, "server.tools를 못 불러와요")
class KrKeyNewTest(unittest.TestCase):
    def run_on(self, raw: bytes) -> tuple[bytes, str]:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ai-keys.env"
            p.write_bytes(raw)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(kr_key_new(p), 0)
            return p.read_bytes(), out.getvalue()

    def new_key(self, data: bytes) -> str:
        lines = [x for x in data.decode("utf-8-sig").splitlines() if x.startswith("KR_API_KEY=")]
        self.assertTrue(lines)
        return lines[0].split("=", 1)[1]

    def test_replaces_line_and_keeps_crlf_bom(self):
        old = "kr-" + "a" * 24
        raw = ("﻿# 키\r\nANTHROPIC_API_KEY=sk-x\r\nKR_API_KEY=" + old + "\r\nMOCK_GENERATION=0\r\n").encode("utf-8")
        data, out = self.run_on(raw)
        key = self.new_key(data)
        self.assertNotEqual(key, old)
        self.assertRegex(key, r"^kr-[A-Za-z0-9]{24}$")
        self.assertEqual(data, raw.replace(old.encode(), key.encode()))  # 나머지 바이트는 그대로
        self.assertNotIn(key, out)  # 화면에 키가 안 나온다
        self.assertNotIn(old, out)

    def test_appends_when_missing_and_skips_comments(self):
        raw = b"# KR_API_KEY=\nELEVENLABS_API_KEY=e"
        data, _ = self.run_on(raw)
        text = data.decode()
        self.assertTrue(text.startswith("# KR_API_KEY=\nELEVENLABS_API_KEY=e\nKR_API_KEY=kr-"))
        self.assertTrue(text.endswith("\n"))

    def test_all_active_lines_change(self):
        raw = b"KR_API_KEY=one\nKR_API_KEY = two\n"
        data, _ = self.run_on(raw)
        lines = data.decode().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], lines[1])
        self.assertNotIn("one", data.decode())


if __name__ == "__main__":
    unittest.main()
