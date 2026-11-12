"""가사 만들기: 아이가 말한 단어 + 고정 서술 틀. 아이가 말한 단어만 바뀐다."""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

from .text import Noun, Ono

# 단어 종류별 서술 틀 (각 5개, 총 20개). {w:받침 있을 때/없을 때} 로 조사를 자동으로 고른다.
TEMPLATES = {
    "object": [
        "{w:이/가} 데굴데굴 굴러가요",
        "반짝반짝 {w:을/를} 찾았어요",
        "{w} 하나 들고 신나게 가요",
        "{w:이/가} 퐁당퐁당 춤을 춰요",
        "{w:과/와} 함께 노래해요",
    ],
    "animal": [
        "{w:이/가} 깡충깡충 따라와요",
        "{w:아/야} {w:아/야} 어디 가니",
        "꼬리 흔드는 {w:을/를} 만났어요",
        "{w:이/가} 우리 차에 탔어요",
        "{w:과/와} 같이 춤을 춰요",
    ],
    "person": [
        "{w:이랑/랑} 손잡고 노래해요",
        "{w:이/가} 웃으면 나도 웃어요",
        "사랑하는 {w} 최고예요",
        "{w:이/가} 불러 주는 노래",
        "{w:과/와} 함께 부릉부릉",
    ],
    "place": [
        "{w:으로/로} 부릉부릉 출발해요",
        "{w}에 가면 뭐가 있을까",
        "{w}까지 노래하며 가요",
        "신나는 {w}, 우리가 간다",
        "{w}에서 만나요 랄랄라",
    ],
}
# 단어가 4개보다 적을 때 채우는 주행 문장
DRIVE_LINES = {
    "shuffle": ["멈췄다가 다시 출발해요", "빨간불엔 잠깐 쉬어 가요", "초록불엔 부릉부릉 출발", "창밖으로 바람이 불어와요"],
    "four_on_floor": ["쌩쌩 달려요 우리 차", "바람처럼 달려가요", "창밖 나무가 휙휙 지나가요", "길을 따라 쭉쭉 달려요"],
}
PLACEHOLDER = re.compile(r"\{w(?::([^}]+))?\}")


def has_jong(word: str) -> bool:
    c = ord(word[-1]) - 0xAC00
    return 0 <= c < 11172 and c % 28 != 0


def jong_is_rieul(word: str) -> bool:
    c = ord(word[-1]) - 0xAC00
    return 0 <= c < 11172 and c % 28 == 8


def josa(word: str, form: str) -> str:
    """form은 '받침 있을 때/없을 때' 순서. 예: '이/가', '을/를', '과/와', '아/야', '으로/로'."""
    with_jong, without = form.split("/")
    if form == "으로/로":
        return word + ("로" if (not has_jong(word) or jong_is_rieul(word)) else "으로")
    return word + (with_jong if has_jong(word) else without)


def fill(template: str, word: str) -> str:
    return PLACEHOLDER.sub(lambda m: josa(word, m.group(1)) if m.group(1) else word, template)


def syllables(text: str) -> int:
    return sum(1 for ch in text if "가" <= ch <= "힣")


@dataclass
class Lyrics:
    title: str
    verse: list[str]
    hook: list[str]  # TTS가 부를 후렴 두 줄 (의성어 포함)
    hook_tail: list[str]  # 섞기 버전에서 TTS가 부를 후렴 뒷부분 (의성어는 아이 목소리)
    words: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)


def build(nouns: list[Noun], onos: dict[str, Ono], pattern: str) -> Lyrics:
    lines: list[str] = []
    used: list[str] = []
    for i, n in enumerate(nouns[:4]):
        cands = TEMPLATES[n.category]
        start = (zlib.crc32(n.text.encode("utf-8")) + i) % len(cands)
        for k in range(len(cands)):
            tpl = cands[(start + k) % len(cands)]
            if tpl not in used:
                break
        used.append(tpl)
        lines.append(fill(tpl, n.text))
    fillers = DRIVE_LINES.get(pattern, DRIVE_LINES["four_on_floor"])
    k = 0
    while len(lines) < 4:
        lines.append(fillers[k % len(fillers)])
        used.append("(주행 문장)")
        k += 1
    first = nouns[0].text if nouns else "우리 차"
    title = f"{first} 노래"
    main = onos["main"].text if "main" in onos else "둥둥"
    mel = onos["melody"].text if "melody" in onos else "랄랄라"
    hook = [f"{main} {main}, {title}", f"{mel} 신나는 우리 노래"]
    hook_tail = [title, "신나는 우리 노래"]
    return Lyrics(title, lines, hook, hook_tail, [n.text for n in nouns[:4]], used)
