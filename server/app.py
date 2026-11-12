"""HTTP API (docs/API.md v0.4)."""
from __future__ import annotations

import hmac
import json

import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .audio_io import BadAudio, decode
from .config import (CHUNK_MS, INSTRUMENTS, KEEP_SEC, MAX_MEMBERS, MAX_RECORDING_SEC, MIN_ENROLL_SPEECH_SEC,
                     SHERPA_DIR, SR_IN)
from .jobs import Store
from .live import Member
from .speaker import speech_report


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"code": code, "message": message}, status_code=status)


LOOPBACK = {"127.0.0.1", "::1", "localhost"}
# 임시 공개 주소(cloudflared)나 프록시를 거친 요청에 붙는 머리글. 이게 있으면 같은 PC에서 왔어도 바깥 요청이다
PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "forwarded")


def needs_key(client_host: str | None, headers) -> bool:
    """KR_API_KEY를 정했을 때 이 요청에 키를 요구할지.

    같은 PC에서 바로 온 요청(에뮬레이터 앱의 10.0.2.2도 서버에는 127.0.0.1로 보인다)은 키 없이 받는다.
    share-server.bat의 임시 공개 주소처럼 바깥에서 들어온 요청은 키가 있어야 한다.
    """
    if client_host not in LOOPBACK:
        return True
    return any(h in headers for h in PROXY_HEADERS)


def parse_members(raw, dim: int, log) -> list[Member]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ApiError(400, "BAD_REQUEST", "members는 배열이어야 해요")
    if len(raw) > MAX_MEMBERS:
        raise ApiError(400, "BAD_REQUEST", f"가족은 최대 {MAX_MEMBERS}명이에요")
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ApiError(400, "BAD_REQUEST", "members 항목 형식이 틀렸어요")
        try:
            mid = int(item.get("memberId"))
        except (TypeError, ValueError):
            raise ApiError(400, "BAD_REQUEST", "memberId는 숫자여야 해요")
        if mid in seen:
            raise ApiError(400, "BAD_REQUEST", f"memberId {mid}가 두 번 있어요")
        seen.add(mid)
        inst = str(item.get("instrument") or "DRUM").upper()
        if inst not in INSTRUMENTS:
            log(f"[알림] 모르는 악기 {inst} -> 북 소리로 대신해")
        emb = item.get("embedding")
        vec = None
        if isinstance(emb, list) and emb:
            if len(emb) == dim:
                vec = np.asarray(emb, dtype=np.float32)
                n = float(np.linalg.norm(vec))
                vec = vec / n if n > 0 else None
            else:
                log(f"[알림] memberId {mid}의 목소리 특징 길이({len(emb)})가 모델({dim})과 달라서 등록 안 한 것으로 봐")
        out.append(Member(mid, inst, vec))
    return out


def create_app(store: Store, api_key: str | None = None, log=print) -> FastAPI:
    engine = store.engine
    app = FastAPI(title="차량 리듬 동요 노래 생성 서버", version=__version__)

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError):
        return error(exc.status, exc.code, exc.message)

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        if api_key and path.startswith("/v1/") and path != "/v1/health":
            host = request.client.host if request.client else None
            if needs_key(host, request.headers) and not hmac.compare_digest(
                    request.headers.get("x-api-key", "").encode(), api_key.encode()):
                return error(401, "UNAUTHORIZED", "X-Api-Key가 없거나 틀려요")
        return await call_next(request)

    @app.get("/v1/health")
    async def health():
        return {"ok": True, "version": __version__, "auth": bool(api_key), **engine.status()}

    # ---------- 2. 목소리 특징 ----------
    def _embed(data: bytes) -> dict:
        try:
            x = decode(data)
        except BadAudio as e:
            raise ApiError(400, "BAD_AUDIO", str(e))
        rep = speech_report(x, SHERPA_DIR)
        quality = "ok"
        if rep["speechSec"] < MIN_ENROLL_SPEECH_SEC:
            quality = "too_short"
        elif rep["levelDb"] < -42.0:
            quality = "too_quiet"
        elif rep["snrDb"] < 8.0:
            quality = "noisy"
        emb = engine.encoder.embed(rep["speech"]) if quality == "ok" else None
        if quality == "ok" and emb is None:
            quality = "too_short"
        log(f"[목소리 등록] {len(x) / SR_IN:.1f}초 중 말소리 {rep['speechSec']:.1f}초, 크기 {rep['levelDb']:.0f}dB, "
            f"잡음 대비 {rep['snrDb']:.0f}dB -> {quality} (녹음은 바로 지움)")
        return {"embedding": [round(float(v), 6) for v in emb] if emb is not None else None,
                "model": engine.encoder.name, "speechSec": rep["speechSec"], "quality": quality}

    @app.post("/v1/voices/embed")
    async def voices_embed(audio: UploadFile = File(...)):
        data = await audio.read()
        return await run_in_threadpool(_embed, data)

    # ---------- 3. 녹음 세션 ----------
    async def _json(request: Request) -> dict:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "BAD_REQUEST", "JSON 형식이 틀렸어요")
        if not isinstance(body, dict):
            raise ApiError(400, "BAD_REQUEST", "JSON 객체를 보내 주세요")
        return body

    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request):
        body = await _json(request)
        members = parse_members(body.get("members"), engine.encoder.dim, log)
        s = store.create_session(members)
        log(f"[세션 {s.sid}] 시작 (가족 {len(members)}명, 등록 목소리 {len(s.enrolled)}명)")
        return {"sessionId": s.sid, "chunkMs": CHUNK_MS, "maxRecordingSec": MAX_RECORDING_SEC}

    def _session(sid: str):
        s = store.get_session(sid)
        if s is None:
            raise ApiError(404, "NOT_FOUND", "없는 세션이에요 (이미 지웠거나 10분이 지났어요)")
        return s

    @app.post("/v1/sessions/{sid}/chunks")
    async def add_chunk(sid: str, request: Request, seq: int = Query(..., ge=0)):
        s = _session(sid)
        if s.finished or store.job_for_session(sid) is not None:
            raise ApiError(409, "ALREADY_FINISHED", "이미 완성하기를 부른 세션이에요")
        data = await request.body()
        if s.would_exceed(len(data)):
            raise ApiError(413, "TOO_LONG", f"녹음이 {MAX_RECORDING_SEC // 60}분을 넘었어요")
        return await run_in_threadpool(s.add_chunk, seq, data)

    @app.post("/v1/sessions/{sid}/finish", status_code=202)
    async def finish(sid: str, request: Request):
        # 앱이 앞 응답을 못 받고 다시 부른 경우: 새로 만들지 않고 같은 노래를 돌려준다
        job = store.job_for_session(sid)
        if job is not None:
            log(f"[세션 {sid}] 완성하기 다시 받음 -> 같은 노래 {job.id}")
            return {"songId": job.id, "status": job.status}
        s = _session(sid)
        body = await _json(request)
        job = store.start_job(s, body.get("drive"), body.get("beat"))
        log(f"[세션 {sid}] 완성하기 -> 노래 {job.id}")
        return {"songId": job.id, "status": job.status}

    @app.delete("/v1/sessions/{sid}", status_code=204)
    async def drop_session(sid: str):
        store.drop_session(sid)
        return Response(status_code=204)

    # ---------- 4. 파일 하나로 ----------
    @app.post("/v1/songs", status_code=202)
    async def create_song(recording: UploadFile = File(...), meta: str = Form("{}")):
        try:
            info = json.loads(meta or "{}")
        except json.JSONDecodeError:
            raise ApiError(400, "BAD_REQUEST", "meta JSON 형식이 틀렸어요")
        members = parse_members(info.get("members"), engine.encoder.dim, log)
        data = await recording.read()
        try:
            audio = await run_in_threadpool(decode, data)
        except BadAudio as e:
            raise ApiError(400, "BAD_AUDIO", str(e))
        if len(audio) / SR_IN > MAX_RECORDING_SEC + 1:
            raise ApiError(413, "TOO_LONG", f"녹음이 {MAX_RECORDING_SEC // 60}분을 넘었어요")
        job = await run_in_threadpool(store.start_job_from_audio, audio, members, info.get("drive"), info.get("beat"))
        return {"songId": job.id, "status": job.status}

    # ---------- 5~7. 상태, 음원, 지우기 ----------
    def _job(song_id: str):
        job = store.get_job(song_id)
        if job is None:
            raise ApiError(404, "NOT_FOUND", "없는 노래예요 (이미 지웠거나 10분이 지났어요)")
        return job

    @app.get("/v1/songs/{song_id}")
    async def song_status(song_id: str):
        return _job(song_id).snapshot()

    @app.get("/v1/songs/{song_id}/audio")
    async def song_audio(song_id: str):
        job = _job(song_id)
        if job.status != "DONE" or job.mp3 is None or not job.mp3.exists():
            raise ApiError(404, "NOT_READY", "아직 노래가 완성되지 않았어요")
        return FileResponse(str(job.mp3), media_type="audio/mpeg", filename=f"{song_id}.mp3",
                            headers={"Cache-Control": f"private, max-age={KEEP_SEC}"})

    @app.delete("/v1/songs/{song_id}", status_code=204)
    async def delete_song(song_id: str):
        store.delete_job(song_id)
        return Response(status_code=204)

    return app
