"""녹음 파일들 -> 노래 mp3 + 결과 JSON (앞으로 만들 API 응답과 같은 모양)."""
from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import drive
from . import lyrics as lyrics_mod
from . import text as text_mod
from .arrange import CHORUS_BARS, VERSE_LINE_BARS, Song, render_backing
from .beat import DEFAULT_MODEL, PLACE_OF_SCENARIO, BeatSpec, compose_with_claude, has_claude_key, rule_spec
from .dsp import SR, resample
from .mix import mix, save_mp3
from .synth import MEMBER_INSTRUMENTS, MEMBER_LABELS, SET_LABELS
from .text import Analysis, Word
from .tts import Voice
from .vocals import VARIANTS, Clip, extract_clip, render

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".3gp", ".amr", ".wma", ".mp4", ".webm", ".caf"}
SEASON_LABELS = {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
FEEL_LABELS = {"shuffle": "셔플", "four_on_floor": "곧은 4비트"}
SOURCE_LABELS = {"rule": "규칙", "claude": "Claude"}


def load_audio(path: Path, sr: int) -> np.ndarray:
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        return resample(data.mean(axis=1), file_sr, sr)
    except Exception:
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(str(path), sampling_rate=sr), dtype=np.float32)


@dataclass
class Member:
    member_id: str
    source: Path
    instrument: str
    audio: np.ndarray
    words: list[Word]
    analysis: Analysis


def speaking_rate(words: list[Word]) -> float | None:
    """초당 음절 수. 인식된 단어 구간만 센다."""
    syl = sum(lyrics_mod.syllables(w.text) for w in words)
    dur = sum(max(0.0, w.end - w.start) for w in words)
    return round(syl / dur, 2) if dur >= 0.3 and syl >= 3 else None


def _neighbors(words: list[Word], start: float, end: float) -> tuple[float | None, float | None]:
    prev_end, next_start = None, None
    for w in words:
        if w.end <= start + 1e-3:
            prev_end = w.end
        elif w.start >= end - 1e-3 and next_start is None:
            next_start = w.start
    return prev_end, next_start


def _best_clip(members: dict[str, Member], text: str, prefer_member: str, kind: str) -> Clip | None:
    cands = []
    for m in members.values():
        items = m.analysis.nouns if kind == "noun" else m.analysis.onos
        for it in items:
            if it.text == text:
                whole = (it.frac_end - it.frac_start) > 0.98
                cands.append((0 if m.member_id == prefer_member else 1, 0 if whole else 1, it.start, m, it))
    cands.sort(key=lambda c: c[:3])
    for _, _, _, m, it in cands:
        pe, ns = _neighbors(m.words, it.start, it.end)
        c = extract_clip(m.audio, SR, it.start, it.end, it.frac_start, it.frac_end, pe, ns, it.text, m.member_id)
        if c is not None:
            return c
    return None


def analyze_members(inputs: list[Path], transcriber, kiwi, log: Callable[[str], None]) -> list[Member]:
    members: list[Member] = []
    for i, path in enumerate(inputs):
        mid = path.stem
        inst = MEMBER_INSTRUMENTS[i % len(MEMBER_INSTRUMENTS)]
        log(f"[{mid}] 불러오는 중: {path.name} (악기: {MEMBER_LABELS[inst]})")
        a16 = load_audio(path, 16000)
        a44 = load_audio(path, SR)
        if hasattr(transcriber, "current"):
            transcriber.current = mid
        t0 = time.time()
        words = transcriber.transcribe(a16)
        log(f"  알아들은 말 ({time.time() - t0:.1f}초, 길이 {len(a16) / 16000:.1f}초): "
            f"{' '.join(w.text for w in words) or '(없음)'}")
        an = text_mod.analyze(words, mid, kiwi)
        rate = speaking_rate(words)
        log(f"  명사: {', '.join(n.text for n in an.nouns) or '없음'} / 의성어: {', '.join(o.text for o in an.onos) or '없음'}"
            f" / 말 빠르기: {rate if rate else '-'}음절/초")
        members.append(Member(mid, path, inst, a44, words, an))
    return members


def _time_of_day() -> str:
    h = dt.datetime.now().hour
    return "아침" if 5 <= h < 11 else "낮" if h < 17 else "저녁" if h < 20 else "밤"


def beat_context(prof: drive.DriveProfile, members: list[Member], place: str | None,
                 outside_c: float | None) -> dict:
    """Claude에게 보내는 정보. 아이가 한 말의 내용(명사)은 보내지 않는다."""
    all_words = [w for m in members for w in m.words]
    onos, seen = [], set()
    for m in members:
        for o in m.analysis.onos:
            if (o.text, o.member) not in seen:
                seen.add((o.text, o.member))
                onos.append({"text": o.text, "role": o.role, "memberId": o.member})
    return {
        "drive": {"label": prof.label, "medianKmh": round(prof.median_kmh, 1), "stops": prof.stops,
                  "frequentStops": prof.pattern == "shuffle", "speedBucketBpm": prof.bpm,
                  "events": [{"bar": b + 1, "type": e} for b, e in sorted(prof.events.items())]},
        "place": place or PLACE_OF_SCENARIO.get(prof.scenario, "알 수 없음"),
        "season": prof.season,
        "outsideTempC": outside_c,
        "timeOfDay": _time_of_day(),
        "child": {"syllablesPerSec": speaking_rate(all_words), "onomatopoeia": onos[:8]},
        "members": [{"memberId": m.member_id, "instrument": m.instrument} for m in members],
    }


def make_beats(prof: drive.DriveProfile, members: list[Member], mode: str, model: str, place: str | None,
               outside_c: float | None, log: Callable[[str], None]) -> list[tuple[BeatSpec, str | None]]:
    """mode: auto(키 있으면 Claude) / rule / claude / both. [(악보, Claude 실패 이유)] 를 돌려준다."""
    pairs = [(m.member_id, m.instrument) for m in members]
    base = rule_spec(prof, pairs)
    want_claude = mode in ("claude", "both") or (mode == "auto" and has_claude_key())
    out: list[tuple[BeatSpec, str | None]] = []
    if mode in ("rule", "both") or not want_claude:
        out.append((base, None))
    if want_claude:
        ctx = beat_context(prof, members, place, outside_c)
        log(f"  Claude({model})에게 비트 작곡 요청 중...")
        spec, err = compose_with_claude(ctx, base, [m.member_id for m in members], model)
        if err:
            log(f"  [알림] {err} -> 규칙 비트로 만들게")
            if mode != "both":
                out.append((base, err))
        else:
            log(f"  Claude 비트 ({spec.latency_sec}초): BPM {spec.bpm}, 스윙 {spec.swing}, {spec.key}장조, "
                f"{SET_LABELS[spec.instrument_set]}, 분위기 '{spec.mood}'")
            log(f"    이유: {spec.reason}")
            if spec.fixes:
                log(f"    고친 값: {', '.join(spec.fixes)}")
            out.append((spec, None))
    elif mode == "auto":
        log("  (Claude 키가 없어서 규칙 비트로 만들어. 키를 넣으면 Claude가 작곡해.)")
    return out


def make_songs(members: list[Member], out_dir: Path, voice: Voice | None, scenarios: list[str],
               variants: list[str], season: str | None, log: Callable[[str], None],
               stt_name: str = "", tts_name: str = "", beat_mode: str = "auto",
               claude_model: str = DEFAULT_MODEL, place: str | None = None,
               outside_c: float | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {m.member_id: m for m in members}
    analyses = [m.analysis for m in members]
    nouns = text_mod.pick_nouns(analyses)
    onos = text_mod.pick_onos(analyses)
    noun_clips = [_best_clip(by_id, n.text, n.member, "noun") for n in nouns]
    ono_clips = {k: _best_clip(by_id, o.text, o.member, "ono") for k, o in onos.items()}
    ono_texts = {k: o.text for k, o in onos.items()}
    log(f"가사에 쓸 단어: {', '.join(n.text for n in nouns) or '없음'}")
    log(f"비트에 쓸 의성어: {', '.join(f'{k}={o.text}' for k, o in onos.items()) or '없음'}")
    missing = [n.text for n, c in zip(nouns, noun_clips) if c is None] + \
              [o.text for k, o in onos.items() if ono_clips.get(k) is None]
    if missing:
        log(f"  (잘라 내지 못한 소리: {', '.join(missing)})")
    member_pairs = [(m.member_id, m.instrument) for m in members]

    reports = []
    for scen in scenarios:
        prof = drive.from_scenario(scen, season, outside_c)
        log(f"== {prof.label}: 중앙 속도 {prof.median_kmh:.0f}km/h, 정차 {prof.stops}번, "
            f"{SEASON_LABELS[prof.season]}")
        beats = make_beats(prof, members, beat_mode, claude_model, place, outside_c, log)
        for spec, fallback in beats:
            lyr = lyrics_mod.build(nouns, onos, spec.feel)
            song = Song(spec)
            tag = f"{scen}-{spec.source}" if beat_mode == "both" else scen
            log(f"  [{SOURCE_LABELS[spec.source]} 비트] BPM {spec.bpm}, {FEEL_LABELS[spec.feel]}, {spec.key}장조, "
                f"{SET_LABELS[spec.instrument_set]}, 곡 길이 {song.length:.1f}초")
            stems = render_backing(song, member_pairs)
            files = []
            for v in variants:
                if v in ("B", "C") and voice is None:
                    log(f"  {v} 버전은 TTS가 없어서 건너뜀")
                    continue
                t0 = time.time()
                vox, placed = render(v, song, lyr, noun_clips, ono_clips, ono_texts, voice)
                st = mix(stems, vox, SR, v)
                fname = f"{tag}_{v}_{VARIANTS[v][0]}.mp3"
                save_mp3(st, SR, out_dir / fname)
                files.append({"variant": v, "label": VARIANTS[v][1], "file": fname,
                              "durationSec": round(len(st) / SR, 2),
                              "peakDb": round(float(20 * np.log10(np.max(np.abs(st)) + 1e-9)), 1),
                              "vocalEvents": len(placed)})
                log(f"    {v} ({VARIANTS[v][1]}) -> {fname} ({time.time() - t0:.1f}초)")
            spec_d = spec.as_dict()
            (out_dir / f"{tag}_beat.json").write_text(json.dumps(spec_d, ensure_ascii=False, indent=2), encoding="utf-8")
            report = {
                "songId": f"{dt.datetime.now():%Y%m%d-%H%M%S}-{tag}",
                "title": lyr.title,
                "createdAt": dt.datetime.now().isoformat(timespec="seconds"),
                "drive": {"scenario": scen, "label": prof.label, "medianKmh": round(prof.median_kmh, 1),
                          "stops": prof.stops, "season": prof.season,
                          "events": [{"bar": b + 1, "type": e} for b, e in sorted(prof.events.items())]},
                "beat": {"source": spec.source, "model": spec.model, "latencySec": spec.latency_sec,
                         "bpm": spec.bpm, "swing": spec.swing, "key": spec.key,
                         "instrumentSet": spec.instrument_set, "instrumentStyle": spec.instrument_style,
                         "mood": spec.mood, "reason": spec.reason, "progression": spec.progression,
                         "fills": spec.fills, "fixes": spec.fixes, "fallbackReason": fallback},
                "members": [{"memberId": m.member_id, "instrument": m.instrument, "sourceFile": m.source.name,
                             "transcript": m.analysis.transcript, "syllablesPerSec": speaking_rate(m.words)}
                            for m in members],
                "words": [{"text": n.text, "category": n.category, "memberId": n.member,
                           "start": round(n.start, 2), "end": round(n.end, 2)} for n in nouns],
                "onomatopoeia": [{"slot": k, "text": o.text, "role": o.role, "memberId": o.member,
                                  "start": round(o.start, 2), "end": round(o.end, 2)} for k, o in onos.items()],
                "metaOnlyWords": sorted({w for m in members for w in m.analysis.meta_only}),
                "lyrics": ([{"section": "verse", "bar": b + 1, "text": t} for b, t in zip(VERSE_LINE_BARS, lyr.verse)]
                           + [{"section": "chorus", "bar": b + 1, "text": lyr.hook[i % 2]} for i, b in enumerate(CHORUS_BARS)]
                           + [{"section": "outro", "bar": 15, "text": lyr.title}]),
                "variants": files,
                "engine": {"stt": stt_name, "tts": tts_name, "beat": spec.model or "rule"},
            }
            (out_dir / f"{tag}_response.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(report)
    _summary(out_dir, reports, members)
    return reports


def _summary(out_dir: Path, reports: list[dict], members: list[Member]) -> None:
    lines = ["차량 리듬 동요 샘플", "", "가족 목소리"]
    for m in members:
        lines.append(f"- {m.member_id} ({MEMBER_LABELS[m.instrument]}): {m.analysis.transcript or '(알아들은 말 없음)'}")
    for r in reports:
        d, b = r["drive"], r["beat"]
        lines += ["", f"[{d['label']} / {SOURCE_LABELS[b['source']]} 비트] 제목: {r['title']}",
                  f"  BPM {b['bpm']}, 스윙 {b['swing']}, {b['key']}장조, {SET_LABELS[b['instrumentSet']]}, "
                  f"{SEASON_LABELS[d['season']]}, 분위기: {b['mood']}"]
        if b["reason"]:
            lines.append(f"  비트를 이렇게 정한 이유: {b['reason']}")
        if b["fallbackReason"]:
            lines.append(f"  (Claude 대신 규칙 비트를 쓴 이유: {b['fallbackReason']})")
        for item in r["lyrics"]:
            lines.append(f"  {item['bar']:>2}마디  {item['text']}")
        for f in r["variants"]:
            lines.append(f"  - {f['file']}  ({f['label']}, {f['durationSec']}초)")
    lines += ["", "들어볼 때 볼 것: 가사가 들리는지, 아이 목소리가 살아 있는지, 주행과 이어진 느낌이 드는지."]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_inputs(paths: list[str], input_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        q = Path(p)
        if q.is_dir():
            files += sorted(f for f in q.iterdir() if f.suffix.lower() in AUDIO_EXT)
        elif q.suffix.lower() in AUDIO_EXT and q.exists():
            files.append(q)
    if not files and input_dir and input_dir.exists():
        files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXT)
    return files[:3]
