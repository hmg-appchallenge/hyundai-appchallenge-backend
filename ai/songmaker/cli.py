"""명령줄 입구: python -m songmaker [녹음 파일...]"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.path = path

    def __call__(self, msg: str = "") -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def load_keys(paths: list[Path]) -> list[str]:
    """ai-keys.env의 KEY=값 줄을 환경 변수로 올린다. 값은 화면이나 기록에 남기지 않는다."""
    loaded = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v
                loaded.append(k)
    return loaded


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(prog="songmaker", description="녹음 파일로 차량 리듬 동요 샘플을 만든다")
    ap.add_argument("inputs", nargs="*", help="녹음 파일 (최대 3개, 파일 하나 = 가족 한 명)")
    ap.add_argument("--input-dir", default=str(AI_DIR / "input"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=str(AI_DIR / "models"))
    ap.add_argument("--logs", default=str(AI_DIR / "logs"))
    ap.add_argument("--scenarios", default="city,highway", help="city, highway, suburb 중 쉼표로")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--season", choices=["spring", "summer", "autumn", "winter"], default=None)
    ap.add_argument("--stt-model", default="large-v3-turbo")
    ap.add_argument("--voice", default="F1", help="TTS 목소리 (F1~F5, M1~M5)")
    ap.add_argument("--tts-pitch", type=float, default=3.0, help="TTS 음높이 올리기(반음)")
    ap.add_argument("--no-hint", action="store_true", help="음성 인식에 의성어 힌트를 주지 않음")
    ap.add_argument("--beat", choices=["auto", "rule", "claude", "both"], default="auto",
                    help="비트 작곡: auto(키 있으면 Claude), rule(규칙), claude, both(둘 다 만들어 비교)")
    ap.add_argument("--claude-model", default="claude-haiku-5-5")
    ap.add_argument("--place", default=None, help="장소 분위기 (예: 바닷가, 산길). 정확한 위치는 보내지 않음")
    ap.add_argument("--temp", type=float, default=None, help="바깥 온도(섭씨)")
    ap.add_argument("--selftest", action="store_true", help="TTS로 만든 가짜 목소리로 시험")
    ap.add_argument("--voices", action="store_true", help="TTS 목소리 10개 미리 듣기 파일 만들기")
    ap.add_argument("--download", action="store_true", help="모델만 내려받기")
    args = ap.parse_args(argv)

    stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    log = Log(Path(args.logs) / f"songmaker-{stamp}.log")
    loaded = load_keys([AI_DIR.parent / "ai-keys.env", AI_DIR / "ai-keys.env"])
    if loaded:
        log(f"키 불러옴: {', '.join(loaded)}")
    models = Path(args.models)
    models.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SUPERTONIC_CACHE_DIR", str(models / "supertonic3"))
    try:
        if args.download:
            from .download import download

            download(models, args.stt_model, log)
            log("모델 준비 끝.")
            return 0

        from .tts import SupertonicTTS, Voice

        suffix = "-selftest" if args.selftest else "-voices" if args.voices else ""
        out = Path(args.out) if args.out else AI_DIR / "output" / f"{stamp}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        log(f"결과 폴더: {out}")

        t0 = time.time()
        log(f"TTS 준비 중 (Supertonic-3, 목소리 {args.voice}) ...")
        tts = SupertonicTTS(args.voice, models)
        voice = Voice(tts, pitch=args.tts_pitch)
        log(f"  준비 끝 ({time.time() - t0:.1f}초)")

        if args.voices:
            from .selftest import PREVIEW_LINE, voices_preview

            log(f'목소리 미리 듣기: "{PREVIEW_LINE}" (음높이 +{args.tts_pitch:g}반음)')
            voices_preview(tts, out / "voices_preview.mp3", args.tts_pitch, log)
            log(f"저장: {out / 'voices_preview.mp3'}")
            return 0

        if args.selftest:
            from .selftest import make_fake_inputs

            log("가짜 아이 목소리 만드는 중 ...")
            inputs = make_fake_inputs(tts, out / "selftest_input")
        else:
            from .pipeline import collect_inputs

            inputs = collect_inputs(args.inputs, Path(args.input_dir))
        if not inputs:
            log("녹음 파일이 없어. input 폴더에 넣거나 make-song.bat 위에 끌어다 놓아 줘.")
            return 2
        log(f"녹음 파일 {len(inputs)}개: {', '.join(p.name for p in inputs)}")

        t0 = time.time()
        log(f"음성 인식 준비 중 ({args.stt_model}) ...")
        from .stt import Transcriber

        stt = Transcriber(args.stt_model, models / "whisper", hint=not args.no_hint)
        log(f"  준비 끝 ({time.time() - t0:.1f}초)")
        from kiwipiepy import Kiwi

        from .pipeline import analyze_members, make_songs

        members = analyze_members(inputs, stt, Kiwi(), log)
        make_songs(members, out, voice, [s.strip() for s in args.scenarios.split(",") if s.strip()],
                   [v.strip().upper() for v in args.variants.split(",") if v.strip()], args.season, log,
                   f"faster-whisper/{args.stt_model}", tts.name, args.beat, args.claude_model,
                   args.place, args.temp)
        log("")
        log(f"완성! 결과 폴더: {out}")
        return 0
    except Exception as e:  # 사용자에게 보여 줄 짧은 설명 + 로그에 전체 기록
        log(f"[멈춤] {type(e).__name__}: {e}")
        log.f.write(traceback.format_exc())
        log.f.flush()
        log(f"자세한 기록: {log.path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
