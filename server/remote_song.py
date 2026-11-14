"""외부 AI로 노래 만들기 (ai-keys.env의 MOCK_GENERATION=0일 때).

- 가사: Claude가 아이가 말한 명사·의성어와 실제로 한 말, 주행 정보를 보고 쓴다.
- 노래: ElevenLabs Music이 반주와 보컬을 만든다. 속도(BPM), 계절 악기, 가족 악기, 정차 패턴(셔플)은
  코드에서 정한 스타일로 넘겨서 "주행 데이터가 음악이 된다"는 연결이 그대로 남는다.
- 아이 목소리: 녹음에서 잘라 낸 의성어·단어 조각을 완성곡의 빈 박자에 얹는다 (이 앱만의 차별점).

결과 JSON은 PC 안 엔진(songmaker.session_song)과 같은 모양이라 앱은 바꿀 게 없다.
어느 단계든 실패하면 RemoteError를 올리고, 서버가 PC 안 엔진으로 대신 만든다 (engines.py).
외부로 나가는 것은 글자(단어·가사)뿐이고 녹음 소리는 나가지 않는다.
"""
from __future__ import annotations

import datetime as dt
import email.parser
import email.policy
import io
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .remote_http import request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ELEVEN_URL = "https://api.elevenlabs.io/v1/music/detailed"
LYRICS_MODEL = "claude-sonnet-5-5"
# 앞에서부터 시도한다. 계정에서 못 쓰는 모델이면 다음 것으로 (요청 단계에서 거절돼 돈이 나가지 않는다)
MUSIC_MODELS = ("music_v2_5", "music_v2", "music_v1")
MUSIC_TIMEOUT_SEC = 200.0

# 노래 길이 (0.6.0): 가족이 많이 말할수록 절을 늘린다. 어떤 빠르기에서도 3분을 넘지 않게 여유를 둔다
MAX_SONG_SEC = 180.0
FORM_LIMIT_SEC = 175.0  # 반주가 계획보다 조금 길게 나와도 3분 안에 들게


def song_verses(speech_sec: float) -> int:
    """가족이 말한 시간(초)으로 절 수를 정한다. 15초 미만 2절, 30초 미만 3절, 그 이상 4절."""
    return 2 if speech_sec < 15 else 3 if speech_sec < 30 else 4


def _build_form(verses: int, double: bool, brk: bool) -> list[tuple[str, str, int, int]]:
    long = verses >= 4
    edge = 4 if long else 2
    form = [("Intro", "intro", edge, 0)]
    for i in range(1, verses + 1):
        if brk and i == 3:
            form.append(("Break", "break", 4, 0))
        form.append((f"Verse {i}", "verse", 8, 4))
        twice = double and (long or i == verses)
        form.append(("Chorus", "chorus", 8 if twice else 4, 4 if twice else 2))
    form.append(("Outro", "outro", edge, 1))
    return form


def song_form(verses: int, bpm: float, limit_sec: float = FORM_LIMIT_SEC) -> list[tuple[str, str, int, int]]:
    """노래 구성 (이름, 앱 가사 상자 section, 마디 수, 가사 줄 수). 가사 한 줄 = 2마디.
    2절: 전주·1절·후렴·2절·후렴·후주 28마디 (110 BPM에서 1분 1초)
    3절: 3절과 후렴을 더하고 마지막 후렴은 두 번 44마디 (1분 36초)
    4절: 후렴은 매번 두 번, 3절 앞에 간주, 전주·후주를 4마디로 76마디 (2분 46초)
    limit_sec를 넘으면 후렴 반복, 간주, 마지막 절 순서로 덜어 낸다 (80 BPM 4절은 56마디 2분 48초)."""
    bar = 240.0 / bpm
    verses = max(2, min(4, verses))
    double = brk = verses >= 3
    while True:
        form = _build_form(verses, double, brk and verses >= 4)
        if sum(b for _, _, b, _ in form) * bar <= limit_sec or (verses <= 2 and not double):
            return form
        if double:
            double = False
        elif brk and verses >= 4:
            brk = False
        else:
            verses -= 1


def form_seconds(form: list[tuple[str, str, int, int]], bpm: float) -> float:
    return sum(b for _, _, b, _ in form) * 240.0 / bpm


FORM = song_form(2, 110)  # 기본(짧은) 구성

SEASON_KO = {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
SEASON_STYLES = {"spring": ["ukulele", "light spring mood"], "summer": ["steel drum", "sunny summer mood"],
                 "autumn": ["acoustic guitar", "warm autumn mood"], "winter": ["glockenspiel", "sleigh bells", "cozy winter mood"]}
INSTRUMENT_STYLES = {"DRUM": "hand drum", "TAMBOURINE": "tambourine", "SHAKER": "shaker", "BELL": "hand bells",
                     "UKULELE": "ukulele", "STEELDRUM": "steel drum", "GUITAR": "acoustic guitar", "XYLOPHONE": "xylophone",
                     "WOODBLOCK": "wood block", "HARP": "harp", "PIANO": "piano", "MARACAS": "maracas"}
BEAT_STYLES = {2: "relaxed gentle beat", 4: "simple four-beat rhythm", 8: "bouncy eighth-note beat",
               16: "energetic sixteenth-note groove"}
NEGATIVE_STYLES = ["explicit lyrics", "dark", "sad", "minor key", "heavy metal", "distorted guitar", "rap",
                   "screaming", "spoken word"]


class RemoteError(Exception):
    """외부 생성 실패. 서버는 이걸 받으면 PC 안 엔진으로 대신 만든다."""


def remote_enabled() -> bool:
    """MOCK_GENERATION=0이고 ElevenLabs 키가 있을 때만 외부로 만든다 (그 밖에는 돈이 드는 호출을 안 한다)."""
    return os.environ.get("MOCK_GENERATION", "1").strip() == "0" and bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())


# ---------- Claude ----------
def claude_message(prompt: str, system: str | None = None, max_tokens: int = 1500,
                   model: str | None = None, timeout: float = 60.0) -> tuple[str, str]:
    """Messages API를 한 번 부른다. (답 글자, 실제 모델) — 패키지 없이 HTTP로 부른다."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RemoteError("ANTHROPIC_API_KEY가 없어요")
    body = {"model": model or os.environ.get("CLAUDE_LYRICS_MODEL") or LYRICS_MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    status, _h, data = request("POST", ANTHROPIC_URL, headers, json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout)
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        obj = None
    if status != 200 or not isinstance(obj, dict):
        err = obj.get("error", {}).get("message") if isinstance(obj, dict) else data[:200].decode("utf-8", "replace")
        raise RemoteError(f"Claude HTTP {status}: {err}")
    text = "".join(b.get("text", "") for b in obj.get("content", []) if b.get("type") == "text")
    return text, str(obj.get("model", body["model"]))


LYRICS_SYSTEM_TEMPLATE = """너는 차 뒷좌석의 3~7세 아이와 가족을 위한 동요 작사가야.
transcript는 가족이 녹음 중에 실제로 한 말이야 (times는 그 말을 한 횟수). 이 말을 거의 그대로 엮어서 한국어 동요 가사를 만들고, 정해진 JSON으로만 답해.

규칙
1. 가사에 쓰는 낱말의 90% 이상은 transcript에 있는 낱말이어야 해. 순서 바꾸기, 되풀이, 줄이기, 조사와 어미를 리듬에 맞게 바꾸기는 괜찮아. 새로 넣는 낱말은 10% 이하로, 이어 주는 말이나 "랄랄라" 같은 소리 정도만 써.
2. 많이 한 말(times가 큰 말)과 신나는 의성어는 후렴(chorus)에 넣어. main과 melody(의성어)는 후렴에 꼭 그대로 넣어.
3. words(들린 명사)는 모두 가사에 한 번 이상 넣어.
4. 잘못 알아들은 것 같은 이상한 말, 무섭거나 슬픈 말, 다치는 말, 어른 말, 욕, 상표는 빼. 사람 이름은 빼거나 "친구"로 바꿔. 엄마·아빠 같은 가족 호칭은 괜찮아.
5. 한 줄은 2마디에 부른다. 줄마다 공백과 문장부호를 빼고 maxSyllables 글자 이하로. 같은 줄을 되풀이해도 좋아.
6. {verse_keys}는 4줄씩, chorus는 2줄, outro는 1줄. chorus는 절마다 똑같이 부른다.{long_hint}
7. title은 12자 이내로, 가족이 한 말에서 골라. 예: "우리 바다 가자"
8. drive(주행 정보)는 분위기 참고용이야. 거기 있는 말을 가사에 새로 넣지는 마.
9. moodStyles는 노래 분위기를 영어 태그 2~3개로 (예: "bouncy", "warm", "sunny"). 장르, 악기, 빠르기는 쓰지 마.

JSON 모양: {shape}
JSON 말고 다른 글은 쓰지 마."""


def lyrics_system(verses: int = 2) -> str:
    """가사 규칙. 절 수(verse1~verseN)에 맞춰 6번 규칙과 JSON 모양을 바꾼다."""
    keys = [f"verse{i}" for i in range(1, verses + 1)]
    verse_keys = "과 ".join(keys) if verses == 2 else ", ".join(keys)
    long_hint = (" 절마다 transcript의 다른 부분을 주로 쓰고, 말이 모자라면 앞 절의 줄을 되풀이해도 돼."
                 if verses > 2 else "")
    shape = {"title": ""}
    for i, k in enumerate(keys):
        shape[k] = ["", "", "", ""]
        if i == 0:
            shape["chorus"] = ["", ""]
    shape.update({"outro": "", "moodStyles": [""]})
    return (LYRICS_SYSTEM_TEMPLATE.replace("{verse_keys}", verse_keys).replace("{long_hint}", long_hint)
            .replace("{shape}", json.dumps(shape, ensure_ascii=False)))


LYRICS_SYSTEM = lyrics_system(2)

_HANGUL_OR_WORD = re.compile(r"[가-힣A-Za-z0-9]")


def syllables(line: str) -> int:
    return len(_HANGUL_OR_WORD.findall(line))


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON이 없음")
    return json.loads(m.group(0))


@dataclass
class Lyrics:
    title: str
    verses: list[list[str]]  # 절마다 4줄 (2~4절)
    chorus: list[str]
    outro: str
    mood: list[str] = field(default_factory=list)
    source: str = "template"
    reuse: float | None = None  # 가사 낱말 중 실제로 한 말에 있던 비율

    def lines(self) -> list[str]:
        out: list[str] = []
        for verse in self.verses:
            out += verse + self.chorus
        return out + [self.outro]


def _clean_line(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().strip('"').strip()


def validate_lyrics(data: dict, nouns: list[str], main: str, melody: str, max_syll: int,
                    verses: int = 2) -> Lyrics:
    from songmaker.text import fit_title, is_blocked

    def lines(key: str, n: int) -> list[str]:
        raw = data.get(key)
        out = [_clean_line(x) for x in raw] if isinstance(raw, list) else []
        out = [x for x in out if x][:n]
        if len(out) < n:
            raise ValueError(f"{key} 줄 수가 모자람")
        return out

    vs = [lines(f"verse{i}", 4) for i in range(1, verses + 1)]
    ch = lines("chorus", 2)
    outro = _clean_line(data.get("outro"))
    title = fit_title(_clean_line(data.get("title")))
    if not outro or not title:
        raise ValueError("outro나 title이 비었음")
    every = [line for v in vs for line in v] + ch + [outro, title]
    for line in every:
        if syllables(line) > max_syll + 4:
            raise ValueError(f"줄이 너무 김: {line}")
        if any(is_blocked(w) for w in re.findall(r"[가-힣]+", line)):
            raise ValueError("쓰면 안 되는 말이 있음")
    text = " ".join(every)
    missing = [n for n in nouns if n not in text]
    if missing:
        raise ValueError(f"들린 명사가 빠짐: {', '.join(missing)}")
    if main and main[:2] not in " ".join(ch):
        raise ValueError(f"후렴에 의성어 {main}가 없음")
    mood = [str(m)[:24] for m in (data.get("mood") or data.get("moodStyles") or []) if isinstance(m, str)][:3]
    return Lyrics(title, vs, ch, outro, mood, "claude")


# 낱말 비교에 쓰는 품사 (조사·어미는 빼서 "가자"와 "가요"를 같은 말로 본다)
CONTENT_TAGS = ("NNG", "NNP", "NNB", "NP", "NR", "VV", "VA", "MAG", "MAJ", "IC", "XR", "MM", "SN", "SL")


def content_units(text: str, kiwi) -> list[str]:
    # 어절마다 따로 분석한다 (붙여서 분석하면 "강아지 강아지"를 "강/아지 강/아"처럼 이상하게 자르기도 해서)
    out = []
    for eojeol in re.findall(r"[가-힣A-Za-z0-9]+", text):
        for t in kiwi.tokenize(eojeol):
            if t.tag.split("-")[0] in CONTENT_TAGS:
                out.append(t.form)
    return out


def reuse_ratio(lyr: Lyrics, said_units: set[str], kiwi) -> tuple[float, list[str]]:
    """가사 낱말(되풀이 포함) 중 실제로 한 말에 있던 비율과, 없던 낱말 목록."""
    units = content_units(" ".join(lyr.lines()), kiwi)
    if not units:
        return 0.0, []
    new = [u for u in units if u not in said_units]
    return 1.0 - len(new) / len(units), list(dict.fromkeys(new))


def write_lyrics(context: dict, nouns: list[str], main: str, melody: str, max_syll: int, log,
                 said_units: set[str] | None = None, kiwi=None, need: float = 0.9,
                 floor: float = 0.7, verses: int = 2) -> tuple[Lyrics | None, str]:
    """Claude로 가사를 받는다. 실제로 한 말을 need 비율 이상 다시 쓰게 하고, 두 번 해도 모자라면
    floor 이상인 것 중 가장 나은 것을 쓴다. 그것도 없으면 (None, 이유)."""
    reason = ""
    base = json.dumps(context, ensure_ascii=False)
    prompt = base
    best: Lyrics | None = None
    t0 = time.time()
    model = ""
    system = lyrics_system(verses)
    for _attempt in range(2):
        try:
            text, model = claude_message(prompt, system, max_tokens=1200 + 300 * (verses - 2), timeout=60)
            lyr = validate_lyrics(_parse_json(text), nouns, main, melody, max_syll, verses)
        except RemoteError as e:
            return None, str(e)  # 키·네트워크 문제는 다시 해도 같다
        except (ValueError, json.JSONDecodeError) as e:
            reason = str(e)
            prompt = base + f"\n\n앞 답이 규칙에 맞지 않았어 ({reason}). 규칙을 다시 지켜서 JSON만 답해."
            continue
        lyr.source = f"claude:{model}"
        if said_units and kiwi is not None:
            lyr.reuse, new = reuse_ratio(lyr, said_units, kiwi)
            if lyr.reuse < need:
                if best is None or (lyr.reuse or 0) > (best.reuse or 0):
                    best = lyr
                reason = f"녹음 말 {lyr.reuse:.0%} (목표 {need:.0%})"
                prompt = (base + f"\n\n앞 가사는 녹음에 없는 낱말이 너무 많아 ({reason}). 녹음에 없던 낱말: "
                          + ", ".join(new[:15]) + ". 이 낱말들을 transcript에 있는 말로 바꿔서 JSON만 다시 답해.")
                continue
        best = lyr
        break
    if best is None or (best.reuse is not None and best.reuse < floor):
        return None, reason or "가사를 못 받음"
    share = f", 녹음 말 {best.reuse:.0%}" if best.reuse is not None else ""
    log(f"가사: Claude {model} ({time.time() - t0:.1f}초) 제목 '{best.title}'{share}")
    return best, ""


def build_transcript(utterances, member_order: list[int], limit_chars: int = 1400) -> list[dict]:
    """가족이 한 말을 시간 순서대로. 이름 대신 가족1·가족2로, 같은 말은 횟수로 묶는다.
    잘못 들었을 가능성이 큰 낱말(확신도 0.35 미만)과 막는 말은 뺀다."""
    from songmaker.text import is_blocked

    order: list[tuple[str, str]] = []
    count: dict[tuple[str, str], int] = {}
    for u in sorted(utterances, key=lambda x: x.start):
        words = [w for w in (u.words or []) if getattr(w, "prob", 1.0) >= 0.35]
        kept = [w.text.strip() for w in words if w.text.strip()
                and not is_blocked(re.sub(r"[^가-힣]", "", w.text))]
        said = re.sub(r"\s+", " ", " ".join(kept)).strip()
        if not said:
            continue
        who = f"가족{member_order.index(u.member_id) + 1}" if u.member_id in member_order else "모르는 사람"
        key = (who, said)
        if key not in count:
            order.append(key)
            count[key] = 0
        count[key] += 1
    out, total = [], 0
    for who, said in order:
        total += len(said)
        if total > limit_chars:
            break
        out.append({"who": who, "said": said, "times": count[(who, said)]})
    return out


def template_lyrics(nouns, main: str, melody: str, title: str, spec) -> Lyrics:
    """Claude를 못 쓸 때: PC 안 엔진과 같은 서술 틀 가사."""
    from songmaker.session_song import FormSong, build_lines

    lines = build_lines(FormSong(spec), nouns, main, melody, title)
    verses = [ln.text for ln in lines if ln.kind == "verse"]
    chorus = [ln.text for ln in lines if ln.kind in ("chorus1", "chorus2")][:2]
    outro = next((ln.text for ln in lines if ln.kind == "outro"), title)
    return Lyrics(title, [verses[:4], verses[4:8] or verses[:4]], chorus, outro, [], "template")


# ---------- ElevenLabs ----------
@dataclass
class Section:
    name: str
    kind: str
    bars: int
    lines: list[str]
    start_ms: int = 0
    dur_ms: int = 0


def plan_sections(lyr: Lyrics, bpm: int, form: list[tuple[str, str, int, int]] | None = None) -> list[Section]:
    bar_ms = 240000.0 / bpm
    form = form or song_form(len(lyr.verses), bpm)
    texts = {"Intro": [], "Break": [], "Chorus": lyr.chorus, "Outro": [lyr.outro]}
    for i, verse in enumerate(lyr.verses, 1):
        texts[f"Verse {i}"] = verse
    out, t = [], 0
    for name, kind, bars, n in form:
        lines = list(texts.get(name, []))
        if kind == "chorus" and lines and n > len(lines):
            lines = (lines * (n // len(lines) + 1))[:n]  # 두 번 부르는 후렴
        dur = int(round(bars * bar_ms))
        out.append(Section(name, kind, bars, lines, t, dur))
        t += dur
    return out


def music_styles(bpm: int, beat_type: int, swing: bool, season: str, instruments: list[str], mood: list[str]) -> list[str]:
    styles = ["Korean children's song", "nursery rhyme", "cheerful", "bright", f"{bpm} BPM", "major key",
              "sweet clear female vocal", "kids choir backing vocals", BEAT_STYLES.get(beat_type, "bouncy beat"),
              "shuffle groove" if swing else "steady straight rhythm"]
    styles += SEASON_STYLES.get(season, [])
    for inst in instruments:
        s = INSTRUMENT_STYLES.get(inst)
        if s and s not in styles:
            styles.append(s)
    styles += [m for m in mood if m not in styles]
    return styles[:18]


LOCAL_STYLES = {"intro": ["instrumental intro", "no vocals"], "verse": ["verse", "light arrangement"],
                "chorus": ["catchy chorus", "kids choir", "hand claps", "fuller arrangement"],
                "outro": ["outro", "gentle ending", "final chord"],
                "break": ["instrumental break", "no vocals", "hand claps"]}


def eleven_body(model: str, sections: list[Section], styles: list[str]) -> dict:
    if model == "music_v1":
        plan = {"positive_global_styles": styles, "negative_global_styles": NEGATIVE_STYLES,
                "sections": [{"section_name": s.name, "positive_local_styles": LOCAL_STYLES[s.kind],
                              "negative_local_styles": ["vocals"] if s.kind in ("intro", "break") else [],
                              "duration_ms": s.dur_ms, "lines": s.lines} for s in sections]}
        return {"model_id": model, "composition_plan": plan, "respect_sections_durations": True, "with_timestamps": True}
    chunks = []
    core = styles[:6]
    for i, s in enumerate(sections):
        body = "\n".join(s.lines) if s.lines else "{instrumental}"
        chunks.append({"text": f"[{s.name}]\n{body}", "duration_ms": s.dur_ms,
                       "positive_styles": styles if i == 0 else core + LOCAL_STYLES[s.kind],
                       "negative_styles": NEGATIVE_STYLES + (["vocals"] if s.kind in ("intro", "break") else [])})
    return {"model_id": model, "composition_plan": {"chunks": chunks}, "with_timestamps": True}


def _parse_multipart(ctype: str, data: bytes) -> tuple[dict, bytes | None]:
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
        b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + data)
    meta: dict = {}
    audio = None
    for part in msg.iter_parts():
        pt = part.get_content_type()
        payload = part.get_payload(decode=True) or b""
        if pt == "application/json" or payload[:1] == b"{":
            try:
                meta.update(json.loads(payload.decode("utf-8")))
                continue
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        if pt.startswith("audio/") or pt == "application/octet-stream" or len(payload) > 10000:
            audio = payload
    return meta, audio


def compose_music(sections: list[Section], styles: list[str], log) -> tuple[bytes, dict, str, str]:
    """(mp3, 메타데이터, 쓴 모델, 노래 id)."""
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise RemoteError("ELEVENLABS_API_KEY가 없어요")
    first = os.environ.get("ELEVENLABS_MUSIC_MODEL", "").strip()
    models = ([first] if first else []) + [m for m in MUSIC_MODELS if m != first]
    last_err = ""
    for model in models:
        body = json.dumps(eleven_body(model, sections, styles), ensure_ascii=False).encode("utf-8")
        t0 = time.time()
        status, hdr, data = request("POST", ELEVEN_URL + "?output_format=mp3_44100_192",
                                    {"xi-api-key": key, "Content-Type": "application/json"}, body, MUSIC_TIMEOUT_SEC)
        if status == 200:
            ctype = hdr.get("content-type", "")
            if ctype.startswith("multipart/"):
                meta, audio = _parse_multipart(ctype, data)
            else:
                meta, audio = {}, data
            if not audio:
                raise RemoteError("ElevenLabs 응답에 소리가 없어요")
            song_id = next((v for k, v in hdr.items() if "song" in k and "id" in k), "")
            cost = {k: v for k, v in hdr.items() if "cost" in k or "credit" in k or "character" in k}
            log(f"노래: ElevenLabs {model} ({time.time() - t0:.1f}초, {len(audio) / 1e6:.2f}MB)"
                + (f" {cost}" if cost else ""))
            return audio, meta, model, song_id
        detail = data[:400].decode("utf-8", "replace")
        last_err = f"{model} HTTP {status}: {detail}"
        log(f"ElevenLabs {model} 거절: HTTP {status} {detail[:200]}")
        if status in (401, 429) or status >= 500:
            break  # 키 문제·한도·서버 오류는 다른 모델로 해도 같다
    raise RemoteError(f"ElevenLabs 실패: {last_err}")


# ---------- 소리 다루기 ----------
def decode_stereo(mp3: bytes, sr: int) -> np.ndarray:
    from songmaker.dsp import resample

    try:
        import soundfile as sf

        x, file_sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
    except Exception:
        import av

        frames = []
        with av.open(io.BytesIO(mp3)) as box:
            rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
            for frame in box.decode(audio=0):
                for f in rs.resample(frame):
                    frames.append(f.to_ndarray())
        if not frames:
            raise RemoteError("ElevenLabs 소리를 읽을 수 없어요")
        x, file_sr = np.concatenate(frames, axis=1).T.astype(np.float32), sr
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if file_sr != sr:
        x = np.stack([resample(x[:, 0], file_sr, sr), resample(x[:, 1], file_sr, sr)], axis=1)
    return x.astype(np.float32)


def onset_curve(mono: np.ndarray, sr: int, hop: int = 441) -> np.ndarray:
    """100 Hz 간격의 소리 시작 세기 (스펙트럼이 갑자기 커지는 정도)."""
    n_fft = 2048
    if len(mono) < n_fft:
        return np.zeros(1, np.float32)
    frames = 1 + (len(mono) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    mag = np.abs(np.fft.rfft(mono[idx] * np.hanning(n_fft)[None, :], axis=1))
    mag = np.log1p(mag * 10.0)
    flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux -= np.convolve(flux, np.ones(15) / 15, mode="same")
    return np.maximum(flux, 0.0).astype(np.float32)


def beat_grid(mono: np.ndarray, sr: int, bpm: float) -> tuple[float, float]:
    """(첫 박 시각 초, 실제 BPM). 요청한 BPM을 믿되 4% 넘게 다르면 들리는 빠르기를 쓴다."""
    curve = onset_curve(mono[: int(40 * sr)], sr)
    fps = sr / 441.0
    if curve.size < 200:
        return 0.0, bpm
    ac = np.correlate(curve - curve.mean(), curve - curve.mean(), mode="full")[curve.size - 1:]
    lo, hi = int(fps * 60 / 150), int(fps * 60 / 66)
    expect = fps * 60.0 / bpm
    lags = np.arange(lo, min(hi, ac.size - 1))
    if lags.size:
        weights = np.exp(-0.5 * ((lags - expect) / (0.06 * expect)) ** 2)
        best = int(lags[np.argmax(ac[lags] * (0.5 + weights))])
        lag = float(best)
        if 1 <= best < ac.size - 1:  # 꼭짓점을 포물선으로 다듬어 빠르기를 더 정확히
            y0, y1, y2 = ac[best - 1], ac[best], ac[best + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                lag = best + 0.5 * (y0 - y2) / den
        found = 60.0 * fps / lag
        if abs(found - bpm) / bpm > 0.04 and 0.5 < found / bpm < 2.0:
            bpm = round(float(found), 1)
    period = fps * 60.0 / bpm
    beats = np.arange(0, curve.size - 1, period)
    best_phase, best_score = 0.0, -1.0
    for ph in np.arange(0.0, period, 0.5):
        pos = (beats + ph).astype(int)
        pos = pos[pos < curve.size]
        score = float(curve[pos].sum())
        if score > best_score:
            best_phase, best_score = ph, score
    # 분석 창의 가운데(1024샘플)를 그 프레임의 시각으로 본다
    first = (best_phase * 441.0 + 1024.0) / sr
    return first % (60.0 / bpm), float(bpm)


def find_word_times(meta) -> list[tuple[float, float, str]]:
    """메타데이터 어디에 있든 단어별 시작·끝 시간을 찾는다 (초)."""
    found: list[tuple[float, float, str]] = []

    def walk(o) -> None:
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o[:3]):
            for x in o:
                txt = x.get("word") or x.get("text") or x.get("token")
                s = x.get("start_ms", x.get("start", x.get("startMs")))
                e = x.get("end_ms", x.get("end", x.get("endMs")))
                if isinstance(txt, str) and isinstance(s, (int, float)) and isinstance(e, (int, float)):
                    ms = "start_ms" in x or "startMs" in x or (isinstance(e, (int, float)) and e > 600)
                    k = 1000.0 if ms else 1.0
                    found.append((float(s) / k, float(e) / k, txt))
            if found:
                return
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(meta)
    return sorted(found)


def overlay_child(track: np.ndarray, sr: int, clips: dict, first_beat: float, bpm: float,
                  sections: list[Section], words: list[tuple[float, float, str]]) -> int:
    """아이 목소리 조각을 박자에 맞춰 얹는다. 노래 가사(단어 시간)와 겹치지 않는 자리만. 얹은 개수를 돌려준다."""
    from songmaker.dsp import add_at

    p = 60.0 / bpm
    main, melody, noun = clips.get("main"), clips.get("melody"), clips.get("noun")
    plan: list[tuple[float, object]] = []
    intro = sections[0]
    t0 = first_beat
    if intro.lines == []:
        for beat, clip in ((0, main), (2, main), (4, melody or main), (6, noun or main)):
            if clip is not None and t0 + beat * p < intro.dur_ms / 1000.0:
                plan.append((t0 + beat * p, clip))

    def free(t: float, length: float) -> bool:
        return all(e < t - 0.08 or s > t + length + 0.08 for s, e, _ in words)

    if words:
        used = 0
        for s in sections[1:]:
            if s.kind not in ("verse", "chorus") or used >= 4:
                continue
            start = s.start_ms / 1000.0
            n_beats = int(round(s.dur_ms / 1000.0 / p))
            for b in range(3, n_beats, 4):  # 마디의 마지막 박 근처
                t = first_beat + start + b * p
                clip = main if used % 2 == 0 else (melody or main)
                if clip is None:
                    continue
                if free(t, len(clip.audio) / sr):
                    plan.append((t, clip))
                    used += 1
                    if used >= 4:
                        break
    # 끝부분은 노래 단어 시간을 알 때만 (모르면 마지막 가사와 겹칠 수 있다)
    outro = sections[-1]
    end_t = first_beat + (outro.start_ms + outro.dur_ms) / 1000.0
    for clip, back in ((melody or main, 2), (main, 1)):
        if clip is None or not words:
            continue
        t = end_t - back * p
        if free(t, len(clip.audio) / sr):
            plan.append((t, clip))
    placed = 0
    for t, clip in plan:
        x = clip.audio.astype(np.float32)
        start = int((t - clip.onset) * sr)
        if start < 0 or start + len(x) > len(track):
            continue
        seg = track[start:start + len(x)]
        local = float(np.sqrt(np.mean(seg ** 2) + 1e-9))
        own = float(np.sqrt(np.mean(x ** 2) + 1e-9))
        g = min(1.2, 0.8 * local / own)
        for ch in range(2):
            add_at(track[:, ch], x, start, g)
        placed += 1
    return placed


# ---------- 한 곡 ----------
LEAD_MS = 150  # 가사 줄을 부르기 시작하기 조금 전에 켠다 (앱이 재생 위치를 띄엄띄엄 읽어도 늦지 않게)


def _sylls(text: str) -> list[str]:
    return [c for c in text if "가" <= c <= "힣" or c.isalnum()]


def align_lines(lines: list[str], words: list[tuple[float, float, str]]) -> list[tuple[float, float] | None]:
    """가사 줄마다 실제로 부른 (시작, 끝) 초. ElevenLabs가 준 단어 시간을 가사 글자와 순서대로 맞춰 본다.
    못 맞춘 줄은 None. 박자표로 나눈 자리는 노래가 앞당기거나 늦게 부르면 어긋나서 이걸 쓴다."""
    lt = [(li, c) for li, t in enumerate(lines) for c in _sylls(t)]
    wt = []
    for s, e, txt in words:
        cs = _sylls(txt)
        step = max(0.0, e - s) / max(1, len(cs))
        wt += [(s + k * step, s + (k + 1) * step, c) for k, c in enumerate(cs)]
    n, m = len(lt), len(wt)
    if not n or not m:
        return [None] * len(lines)
    # 글자 편집 거리. 같은 글자면 0, 다르면 1, 빠지거나 더 부르면 1
    cost = np.zeros((n + 1, m + 1), np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ci = lt[i - 1][1]
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + (ci != wt[j - 1][2]), cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        same = lt[i - 1][1] == wt[j - 1][2]
        if cost[i, j] == cost[i - 1, j - 1] + (not same):
            if same:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif cost[i, j] == cost[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    spans: list[list[float]] = [[] for _ in lines]
    for li_pos, wj in pairs:
        spans[lt[li_pos][0]] += [wt[wj][0], wt[wj][1]]
    out: list[tuple[float, float] | None] = []
    for li, t in enumerate(lines):
        got, total = len(spans[li]) // 2, len(_sylls(t))
        # 한두 글자만 우연히 같은 줄은 믿지 않는다
        out.append((min(spans[li]), max(spans[li])) if got and got >= max(2, total * 0.4) else None)
    return out


def _timings(sections: list[Section], first_beat: float, words: list[tuple[float, float, str]]) -> list[dict]:
    items = []
    for s in sections:
        if not s.lines:
            continue
        slot = s.dur_ms / len(s.lines)
        for i, text in enumerate(s.lines):
            a = first_beat * 1000 + s.start_ms + i * slot  # 박자표로 나눈 자리 (단어 시간이 없을 때)
            items.append({"text": text, "startMs": a, "endMs": a + slot, "section": s.kind})
    sung = align_lines([it["text"] for it in items], words)
    for it, span in zip(items, sung):
        if span is not None:
            it["startMs"] = max(0.0, span[0] * 1000 - LEAD_MS)
            it["endMs"] = span[1] * 1000 + 500  # 마지막 줄은 다 부른 뒤 조금 더 켜 둔다
            it["sung"] = True
        elif words:
            # 글자로 못 맞춘 줄: 예전처럼 박자표 자리 근처에서 처음 부른 단어로 당긴다
            a, slot = it["startMs"], it["endMs"] - it["startMs"]
            near = [w for w in words if a - 600 <= w[0] * 1000 <= a + slot * 0.6]
            if near:
                it["startMs"] = max(a - 600, near[0][0] * 1000 - LEAD_MS)
    # 못 맞춘 줄은 앞뒤 맞춘 줄 사이에서 박자표 자리를 그대로 두되 순서가 꼬이지 않게 한다
    for i in range(1, len(items)):
        if not items[i].get("sung") and items[i]["startMs"] <= items[i - 1]["startMs"]:
            items[i]["startMs"] = items[i - 1]["startMs"] + 300
    for i in range(len(items) - 1, 0, -1):
        if not items[i - 1].get("sung") and items[i - 1]["startMs"] >= items[i]["startMs"]:
            items[i - 1]["startMs"] = max(0.0, items[i]["startMs"] - 300)
    for i, it in enumerate(items):
        it["startMs"] = int(round(it["startMs"]))
        nxt = items[i + 1]["startMs"] if i + 1 < len(items) else it["endMs"]
        it["endMs"] = int(round(max(it["startMs"] + 300, nxt)))
        it.pop("sung", None)
    return items


def _highlights(text: str, nouns, onos: dict) -> list[dict]:
    from songmaker.session_song import _member_int

    out, taken = [], []

    def add(a: int, b: int, member, kind: str) -> None:
        if member is None or any(not (b <= x or a >= y) for x, y in taken):
            return
        taken.append((a, b))
        out.append({"start": a, "end": b, "memberId": member, "kind": kind})

    for n in nouns:
        i = text.find(n.text)
        if i >= 0:
            add(i, i + len(n.text), _member_int(n.member), "word")
    for o in onos.values():
        if o is None:
            continue
        i = text.find(o.text)
        while i >= 0:
            add(i, i + len(o.text), _member_int(o.member), "onomatopoeia")
            i = text.find(o.text, i + len(o.text))
    return sorted(out, key=lambda h: h["start"])


def make_remote_song(audio44: np.ndarray, members, all_members: list[tuple[int, str]], drive: dict | None,
                     beat: dict | None, utterances, progress, log, kiwi=None) -> tuple[np.ndarray, dict]:
    from songmaker.dsp import SR, limit
    from songmaker.session_song import (FORM as LOCAL_FORM, DEFAULT_MAIN, DEFAULT_MELODY, best_clip, make_spec,
                                        make_title, playable)
    from songmaker.drive import season_of
    from songmaker.text import CATEGORY_LABELS, fit_title, pick_nouns, pick_onos

    t_start = time.time()
    progress("WRITE_LYRICS", 0.3)
    analyses = [m.analysis for m in members]
    nouns = pick_nouns(analyses)
    onos = pick_onos(analyses)
    main = onos["main"].text if onos.get("main") else DEFAULT_MAIN
    melody = onos["melody"].text if onos.get("melody") else DEFAULT_MELODY
    if melody == main:
        melody = DEFAULT_MELODY if main != DEFAULT_MELODY else DEFAULT_MAIN
    spec, prof, beat_type, day = make_spec(beat, drive, sum(n for _, n in LOCAL_FORM))
    bpm = int(spec.bpm)
    outside = (drive or {}).get("outsideTempC")
    season = prof.season if prof else season_of(day, float(outside) if isinstance(outside, (int, float)) else None)
    max_syll = 12 if bpm <= 100 else 10
    started = (drive or {}).get("startedAt")
    hour = None
    try:
        hour = dt.datetime.fromisoformat(str(started).replace("Z", "+00:00")).hour if started else None
    except ValueError:
        pass
    if kiwi is None:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
    transcript = build_transcript(utterances, [mid for mid, _ in all_members])
    said_units = set()
    for item in transcript:
        said_units.update(content_units(item["said"], kiwi))
    said_units.update(n.text for n in nouns)
    said_units.update(x for x in (main, melody) if x)
    # 말한 낱말이 적으면(짧은 녹음) 90%를 지키기 어려워서 기준을 낮춘다
    need = 0.9 if len(said_units) >= 8 else 0.75
    # 노래 길이: 가족이 말한 시간만큼 절을 늘린다 (최대 3분)
    spoken = sum(float(getattr(m, "speech_sec", 0.0) or 0.0) for m in members)
    form = song_form(song_verses(spoken), bpm)
    verses = sum(1 for _, kind, _, _ in form if kind == "verse")
    log(f"노래 구성: {verses}절 {sum(b for _, _, b, _ in form)}마디, 약 {form_seconds(form, bpm):.0f}초 "
        f"(가족이 말한 시간 {spoken:.1f}초)")
    context = {
        "transcript": transcript,
        "words": [{"text": n.text, "kind": CATEGORY_LABELS.get(n.category, "사물")} for n in nouns],
        "main": main, "melody": melody,
        "drive": {"bpm": bpm, "season": SEASON_KO.get(season, season),
                  "medianKmh": round(prof.median_kmh) if prof else None,
                  "pattern": ("정차가 잦음" if prof.pattern == "shuffle" else "일정하게 달림") if prof else None,
                  "outsideTempC": (drive or {}).get("outsideTempC"),
                  "timeOfDay": None if hour is None else ("아침" if 5 <= hour < 11 else "낮" if hour < 17 else "저녁" if hour < 21 else "밤")},
        "maxSyllables": max_syll,
        "verses": verses,
    }
    lyr, why = (write_lyrics(context, [n.text for n in nouns], main, melody, max_syll, log,
                             said_units=said_units, kiwi=kiwi, need=need, floor=min(need, 0.7), verses=verses)
                if os.environ.get("ANTHROPIC_API_KEY") else (None, "Claude 키 없음"))
    if lyr is None:
        log(f"가사: 서술 틀로 대신 ({why})")
        lyr = template_lyrics(nouns, main, melody, make_title(nouns, day), spec)
    if len(lyr.verses) != verses:  # 서술 틀 가사는 2절이라 구성도 짧게
        form = song_form(len(lyr.verses), bpm)

    progress("SPLIT_VOICES", 0.4)
    clips = {"main": best_clip(audio44, members, onos["main"].text, onos["main"].member, "ono") if onos.get("main") else None,
             "melody": best_clip(audio44, members, onos["melody"].text, onos["melody"].member, "ono") if onos.get("melody") else None,
             "noun": best_clip(audio44, members, nouns[0].text, nouns[0].member, "noun") if nouns else None}

    progress("ADD_INSTRUMENTS", 0.5)
    known = {mid for mid, _ in all_members}
    spoke = [playable(m.instrument) for m in members if m.member_id in known and m.speech_sec >= 0.5]
    instruments = spoke or [playable(inst) for _mid, inst in all_members]
    styles = music_styles(bpm, beat_type, spec.swing > 0.5, season, instruments, lyr.mood)
    sections = plan_sections(lyr, bpm, form)
    mp3, meta, model, song_id = compose_music(sections, styles, log)

    progress("FINISH", 0.85)
    track = decode_stereo(mp3, SR)
    first_beat, real_bpm = beat_grid(track.mean(axis=1), SR, bpm)
    words = find_word_times(meta)
    placed = overlay_child(track, SR, clips, first_beat, real_bpm, sections, words)
    stereo = limit(track, SR, -1.0, 3.0, 80.0)
    log(f"아이 목소리 조각 {placed}개 얹음 (첫 박 {first_beat:.2f}초, {real_bpm} BPM, 단어 시간 {len(words)}개)")

    slot_onos = {"main": onos.get("main"), "melody": onos.get("melody")}
    lyric_items = _timings(sections, first_beat, words)
    for it in lyric_items:
        it["highlights"] = _highlights(it["text"], nouns, slot_onos)
    from songmaker.session_song import _member_int

    heard_nouns, seen = [], set()
    for a in analyses:
        for n in a.nouns:
            if n.text not in seen:
                seen.add(n.text)
                heard_nouns.append({"text": n.text, "memberId": _member_int(n.member)})
    heard_onos, seen = [], set()
    for a in analyses:
        for o in a.onos:
            if o.text not in seen:
                seen.add(o.text)
                heard_onos.append({"text": o.text, "memberId": _member_int(o.member)})
    speech = {m.member_id: m.speech_sec for m in members}
    song_meta = meta.get("song_metadata") if isinstance(meta, dict) else None
    result = {
        "title": fit_title(lyr.title),
        "durationMs": int(round(len(stereo) / SR * 1000)),
        "bpm": int(round(real_bpm)),
        "beatType": beat_type,
        "firstBeatMs": int(round(first_beat * 1000)),
        "lyrics": lyric_items,
        "words": heard_nouns[:8],
        "onomatopoeia": heard_onos[:8],
        "members": [{"memberId": mid, "instrument": inst, "speechSec": round(float(speech.get(mid, 0.0)), 1)}
                    for mid, inst in all_members],
        "engine": {"generator": f"elevenlabs:{model}", "lyricsBy": lyr.source, "lyricsReuse":
                   None if lyr.reuse is None else round(lyr.reuse, 2), "transcriptLines": len(transcript),
                   "styles": styles, "songId": song_id,
                   "childClips": placed, "season": season, "wordTimes": len(words),
                   # 가사 맞춤 확인용: ElevenLabs가 준 단어 시간 [시작ms, 끝ms, 단어] (앱은 안 씀)
                   "sungWords": [[int(round(s * 1000)), int(round(e * 1000)), t] for s, e, t in words[:160]],
                   "songMetadata": song_meta if isinstance(song_meta, dict) else None,
                   "seconds": round(time.time() - t_start, 1),
                   "drive": ({"medianKmh": round(prof.median_kmh, 1), "stops": prof.stops, "pattern": prof.pattern}
                             if prof else None)},
    }
    return stereo, result
