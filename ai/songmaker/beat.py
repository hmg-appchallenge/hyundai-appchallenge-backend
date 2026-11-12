"""비트 악보(BeatSpec).

규칙으로 만들 수도 있고, Claude API가 주행 정보, 장소, 계절, 아이 말 빠르기를 보고 작곡할 수도 있다.
Claude는 소리를 직접 만들지 않고 악보(BPM, 16칸 드럼 패턴, 화음, 필인, 후렴 멜로디)를 JSON으로 쓴다.
어느 쪽이든 모양이 같아서 같은 엔진(arrange.py)이 연주한다.
Claude 호출이 실패하거나 키가 없으면 규칙 악보로 돌아간다.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from .drive import DriveProfile

DEGREES = ("I", "ii", "iii", "IV", "V", "vi")
DEGREE_INFO = {"I": (0, "maj"), "ii": (2, "min"), "iii": (4, "min"), "IV": (5, "maj"), "V": (7, "maj"), "vi": (9, "min")}
KEY_OFFSET = {"C": 0, "D": 2, "F": 5, "G": 7}
SCALE = (0, 2, 4, 5, 7, 9, 11, 12)  # 장음계 도수 1~8
SETS = ("ukulele", "steel_drum", "acoustic_guitar", "bell")
STYLES = ("strum", "arp8", "calypso", "quarter")
FILL_TYPES = ("snare_roll", "tom_roll", "crash", "open_hat")
DRUM_PARTS = ("kick", "snare", "hat", "openHat", "clap")
SECTIONS = ("intro", "verse", "chorus", "outro")
PROG_LEN = {"intro": 2, "verse": 8, "chorus": 4, "outro": 2}
DEFAULT_STYLE = {"ukulele": "strum", "acoustic_guitar": "arp8", "steel_drum": "calypso", "bell": "quarter"}
PLACE_OF_SCENARIO = {"city": "도심", "highway": "고속도로", "suburb": "교외 국도"}
STEP_RE = re.compile(r"^[xX.]{16}$")
DEFAULT_MODEL = "claude-haiku-5-5"


@dataclass
class BeatSpec:
    bpm: int
    swing: float  # 0 = 곧은 8분음표, 1 = 셋잇단 셔플
    key: str
    instrument_set: str
    instrument_style: str
    progression: dict[str, list[str]]
    drums: dict[str, dict[str, str]]
    fills: list[dict]
    members: dict[str, dict[str, str]]
    hook: list[list[int]]
    mood: str = ""
    reason: str = ""
    source: str = "rule"
    model: str = ""
    latency_sec: float = 0.0
    fixes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def feel(self) -> str:
        return "shuffle" if self.swing >= 0.5 else "four_on_floor"


def _pat(steps, accent=()) -> str:
    steps, accent = set(steps), set(accent)
    return "".join("X" if i in accent else ("x" if i in steps else ".") for i in range(16))


EIGHTHS = range(0, 16, 2)
MEMBER_DEFAULT = {
    "drum": {"verse": _pat([0, 8], [0]), "chorus": _pat([0, 4, 8, 12], [0, 8])},
    "tambourine": {"verse": _pat([4, 12]), "chorus": _pat([2, 4, 6, 10, 12, 14], [4, 12])},
    "shaker": {"verse": _pat(EIGHTHS), "chorus": _pat(list(EIGHTHS) + [7, 15], [2, 6, 10, 14])},
}
DEFAULT_HOOK = [[3, 3, 5, 3], [2, 2, 5, 2], [1, 1, 3, 1], [6, 8, 4, 3]]
DEFAULT_PROG = {"intro": ["I", "V"], "verse": ["I", "IV", "V", "I", "I", "IV", "V", "I"],
                "chorus": ["I", "V", "vi", "IV"], "outro": ["V", "I"]}


def rule_spec(profile: DriveProfile, members: list[tuple[str, str]]) -> BeatSpec:
    """기획 문서의 규칙 그대로: 속도 구간 -> BPM, 정차 잦음 -> 셔플, 가속 -> 필인, 감속 -> 하이햇 오픈."""
    shuffle = profile.pattern == "shuffle"
    if shuffle:
        verse = {"kick": _pat([0, 8, 10], [0, 8]), "snare": _pat([4, 12]), "hat": _pat(EIGHTHS, [0, 4, 8, 12]),
                 "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, clap=_pat([4, 12]))
    else:
        verse = {"kick": _pat([0, 4, 8, 12], [0]), "snare": _pat([4, 12]), "hat": _pat(EIGHTHS),
                 "openHat": _pat([]), "clap": _pat([])}
        chorus = dict(verse, hat=_pat([0, 4, 8, 12]), openHat=_pat([2, 6, 10, 14]), clap=_pat([4, 12]))
    outro = {"kick": _pat([0]), "snare": _pat([8]), "hat": _pat([0, 8]), "openHat": _pat([]), "clap": _pat([])}
    fills = [{"bar": b + 1, "type": "snare_roll" if e == "accel" else "open_hat",
              "why": "가속 구간" if e == "accel" else "감속 구간"} for b, e in sorted(profile.events.items())]
    return BeatSpec(
        bpm=profile.bpm, swing=1.0 if shuffle else 0.0, key="C",
        instrument_set=profile.instrument_set, instrument_style=DEFAULT_STYLE[profile.instrument_set],
        progression={k: list(v) for k, v in DEFAULT_PROG.items()},
        drums={"intro": dict(verse), "verse": verse, "chorus": chorus, "outro": outro},
        fills=fills, members={mid: dict(MEMBER_DEFAULT[inst]) for mid, inst in members},
        hook=[list(h) for h in DEFAULT_HOOK],
        mood="신나는 셔플" if shuffle else "쭉쭉 달리는 비트",
        reason="속도 구간으로 BPM을, 정차 횟수로 셔플 여부를 정한 기본 규칙",
        source="rule",
    )


# ---------- Claude가 쓴 악보 검사 ----------
def _canon_degree(d: str) -> str | None:
    d = str(d).strip()
    for c in DEGREES:
        if d == c or d.upper() == c.upper():
            return c
    return None


def validate(data: dict, base: BeatSpec, member_ids: list[str]) -> BeatSpec:
    """범위를 벗어나거나 모양이 틀린 값은 규칙 악보 값으로 바꾸고 fixes에 적는다."""
    fixes: list[str] = []

    try:
        bpm = int(round(float(data.get("bpm", base.bpm))))
    except (TypeError, ValueError):
        bpm, _ = base.bpm, fixes.append("bpm 형식")
    if not 70 <= bpm <= 140:
        fixes.append(f"bpm {bpm} -> 범위 안으로")
        bpm = min(140, max(70, bpm))
    try:
        swing = min(1.0, max(0.0, float(data.get("swing", base.swing))))
    except (TypeError, ValueError):
        swing, _ = base.swing, fixes.append("swing 형식")

    key = str(data.get("key", base.key)).strip().upper()
    if key not in KEY_OFFSET:
        key, _ = base.key, fixes.append("key")
    inst = str(data.get("instrumentSet", base.instrument_set)).strip().lower()
    if inst not in SETS:
        inst, _ = base.instrument_set, fixes.append("instrumentSet")
    style = str(data.get("instrumentStyle", "")).strip().lower()
    if style not in STYLES:
        style, _ = DEFAULT_STYLE[inst], fixes.append("instrumentStyle")

    prog: dict[str, list[str]] = {}
    raw_prog = data.get("progression") or {}
    for sec in SECTIONS:
        seq = [_canon_degree(d) for d in (raw_prog.get(sec) or [])]
        seq = [d for d in seq if d]
        if not seq:
            prog[sec] = list(base.progression[sec])
            fixes.append(f"progression.{sec}")
            continue
        n = PROG_LEN[sec]
        if len(seq) != n:
            fixes.append(f"progression.{sec} 길이 {len(seq)} -> {n}")
            seq = (seq * n)[:n]
        prog[sec] = seq
    if prog["outro"][-1] != "I":
        prog["outro"][-1] = "I"
        fixes.append("outro는 I로 끝냄")

    drums: dict[str, dict[str, str]] = {}
    raw_drums = data.get("drums") or {}
    for sec in SECTIONS:
        part_in = raw_drums.get(sec) or {}
        out = {}
        for part in DRUM_PARTS:
            s = str(part_in.get(part, "")).strip()
            if not STEP_RE.match(s):
                out[part] = base.drums[sec][part]
                fixes.append(f"drums.{sec}.{part}")
            else:
                out[part] = s
        if sec != "outro" and out["kick"][0] == ".":
            out["kick"] = "X" + out["kick"][1:]
            fixes.append(f"drums.{sec} 첫 칸 킥")
        drums[sec] = out

    fills: list[dict] = []
    seen_bars: set[int] = set()
    for f in data.get("fills") or []:
        try:
            bar = int(f.get("bar"))
        except (TypeError, ValueError, AttributeError):
            continue
        typ = str(f.get("type", "")).strip().lower()
        if typ not in FILL_TYPES or not 3 <= bar <= 14 or bar in seen_bars:
            continue
        seen_bars.add(bar)
        fills.append({"bar": bar, "type": typ, "why": str(f.get("why", ""))[:80]})
        if len(fills) >= 4:
            break

    raw_members = {str(m.get("memberId")): m for m in (data.get("members") or []) if isinstance(m, dict)}
    members: dict[str, dict[str, str]] = {}
    for mid in member_ids:
        m = raw_members.get(mid, {})
        pats = {}
        for sec in ("verse", "chorus"):
            s = str(m.get(sec, "")).strip()
            pats[sec] = s if STEP_RE.match(s) else base.members.get(mid, {}).get(sec, _pat([0, 8]))
            if not STEP_RE.match(s):
                fixes.append(f"members.{mid}.{sec}")
        members[mid] = pats

    hook = data.get("hook")
    ok = isinstance(hook, list) and len(hook) == 4 and all(
        isinstance(b, list) and len(b) == 4 and all(isinstance(d, int) and 1 <= d <= 8 for d in b) for b in hook)
    if not ok:
        hook = [list(h) for h in base.hook]
        fixes.append("hook")

    return BeatSpec(bpm, round(swing, 2), key, inst, style, prog, drums, fills, members, hook,
                    str(data.get("mood", ""))[:20], str(data.get("reason", ""))[:300], "claude", fixes=fixes)


# ---------- Claude 호출 ----------
SYSTEM_PROMPT = """너는 차 뒷좌석의 3~7세 아이를 위한 동요 편곡자야.
주행 정보, 장소, 계절, 아이 목소리 정보를 보고 16마디 비트 악보를 정해진 JSON으로만 답해.

곡 구조: 1~2마디 인트로, 3~10마디 벌스(가사 4줄, 3·5·7·9마디에서 시작), 11~14마디 후렴, 15~16마디 아웃트로.

규칙
1. bpm: drive.speedBucketBpm(속도 구간 기준)을 출발점으로 하고 아이 말 빠르기(child.syllablesPerSec)를 반영해 ±12 안에서 조정해. 보통은 초당 4~5음절이야. 빠르게 말하면 조금 빠르게, 느리면 조금 느리게. 범위는 70~140.
2. swing: 정차가 잦으면(drive.frequentStops가 true) 0.7~1.0 셔플, 일정하게 달리면 0~0.1.
3. drums: 섹션(intro, verse, chorus, outro)마다 kick, snare, hat, openHat, clap을 16칸 문자열로 써. x=보통, X=강세, .=쉼, 한 칸은 16분음표야. intro, verse, chorus는 첫 칸 킥이 꼭 있어야 해. verse는 단순하게, chorus는 clap이나 openHat으로 더 꽉 차게. 아이 노래라 복잡한 당김음은 피하고, outro는 듬성듬성하게 끝내.
4. fills: drive.events의 가속 마디엔 snare_roll 또는 tom_roll, 감속 마디엔 open_hat. 3~14마디만, 최대 4개. 후렴이 시작하는 11마디에 crash를 넣어도 돼. why에는 이유를 짧게.
5. 악기: 계절 악기(spring=ukulele, summer=steel_drum, autumn=acoustic_guitar, winter=bell)가 기본이야. 장소 분위기가 뚜렷하면(예: 바닷가) 바꿔도 돼. instrumentStyle은 strum, arp8, calypso, quarter 중 악기에 어울리는 것.
6. progression: 장조 화음 I, ii, iii, IV, V, vi만 써. intro 2개, verse 8개(마디마다 하나), chorus 4개, outro 2개이고 outro는 V, I로 끝내. 동요답게 밝고 단순하게.
7. members: 가족마다 memberId를 그대로 쓰고 verse, chorus 16칸 패턴을 줘. drum은 1·3박 중심, tambourine은 2·4박, shaker는 8분음표가 기본이고 다른 드럼과 너무 겹치지 않게.
8. hook: 후렴 4마디 멜로디. 마디마다 4분음표 4개를 음계 도수 1~8로 써(1=으뜸음, 8=한 옥타브 위 으뜸음). 그 마디 화음의 구성음 위주로, 아이가 따라 부르기 쉽게 좁은 음역으로.
9. mood: 한국어 8자 이내. reason: 속도, 장소, 계절, 아이 말 빠르기 중 무엇을 어떻게 반영했는지 한국어 한두 문장.
"""

_STEP = {"type": "string", "description": "16칸 문자열. x=보통, X=강세, .=쉼"}
_DRUM_SECTION = {"type": "object", "properties": {p: _STEP for p in DRUM_PARTS},
                 "required": list(DRUM_PARTS), "additionalProperties": False}
_DEGREE_LIST = {"type": "array", "items": {"type": "string", "enum": list(DEGREES)}}
SCHEMA = {
    "type": "object",
    "properties": {
        "bpm": {"type": "integer", "description": "70~140"},
        "swing": {"type": "number", "description": "0=곧게, 1=셋잇단 셔플"},
        "key": {"type": "string", "enum": list(KEY_OFFSET)},
        "instrumentSet": {"type": "string", "enum": list(SETS)},
        "instrumentStyle": {"type": "string", "enum": list(STYLES)},
        "progression": {"type": "object", "properties": {s: _DEGREE_LIST for s in SECTIONS},
                        "required": list(SECTIONS), "additionalProperties": False},
        "drums": {"type": "object", "properties": {s: _DRUM_SECTION for s in SECTIONS},
                  "required": list(SECTIONS), "additionalProperties": False},
        "fills": {"type": "array", "items": {
            "type": "object",
            "properties": {"bar": {"type": "integer"}, "type": {"type": "string", "enum": list(FILL_TYPES)},
                           "why": {"type": "string"}},
            "required": ["bar", "type", "why"], "additionalProperties": False}},
        "members": {"type": "array", "items": {
            "type": "object",
            "properties": {"memberId": {"type": "string"}, "verse": _STEP, "chorus": _STEP},
            "required": ["memberId", "verse", "chorus"], "additionalProperties": False}},
        "hook": {"type": "array", "description": "후렴 4마디, 마디마다 음계 도수 4개(1~8)",
                 "items": {"type": "array", "items": {"type": "integer"}}},
        "mood": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["bpm", "swing", "key", "instrumentSet", "instrumentStyle", "progression", "drums",
                 "fills", "members", "hook", "mood", "reason"],
    "additionalProperties": False,
}


def has_claude_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def compose_with_claude(context: dict, base: BeatSpec, member_ids: list[str],
                        model: str = DEFAULT_MODEL, timeout: float = 30.0) -> tuple[BeatSpec, str | None]:
    """Claude에게 악보를 받아 검사한다. 실패하면 (규칙 악보, 실패 이유)를 돌려준다."""
    if not has_claude_key():
        return base, "Claude 키가 없음 (ai-keys.env의 ANTHROPIC_API_KEY)"
    try:
        import anthropic
    except ImportError:
        return base, "anthropic 패키지가 없음 (setup-ai.bat 다시 실행)"
    t0 = time.time()
    try:
        client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        resp = client.messages.create(
            model=model,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except Exception as e:  # 네트워크, 키, 요청 형식 문제
        return base, f"Claude 호출 실패: {type(e).__name__}: {str(e)[:300]}"
    latency = time.time() - t0
    if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
        return base, f"Claude 응답이 끝까지 오지 않음 (stop_reason={resp.stop_reason})"
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return base, "Claude 응답이 JSON이 아님"
    spec = validate(data, base, member_ids)
    spec.model = model
    spec.latency_sec = round(latency, 2)
    return spec, None
