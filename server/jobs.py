"""세션과 노래 작업을 들고 있는 곳. 아무것도 오래 갖고 있지 않는다 (최대 10분)."""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .audio_io import float_to_pcm16
from .config import CHUNK_MS, JOB_TIMEOUT_SEC, KEEP_SEC, MIN_RECORDING_SEC, SR_IN, TMP_DIR
from .live import LiveSession, Member


class Cancelled(Exception):
    pass


class JobTimeout(Exception):
    pass


class Job:
    def __init__(self, song_id: str):
        self.id = song_id
        self.status = "QUEUED"
        self.step = "FIND_WORDS"
        self.progress = 0.0
        self.result: dict | None = None
        self.error: dict | None = None
        self.mp3: Path | None = None
        self.created = time.time()
        self.done_at: float | None = None
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def update(self, step: str, progress: float) -> None:
        with self.lock:
            if self.status in ("QUEUED", "RUNNING"):
                self.status, self.step, self.progress = "RUNNING", step, max(self.progress, progress)

    def finish(self, result: dict, mp3: Path) -> None:
        with self.lock:
            self.status, self.step, self.progress = "DONE", "FINISH", 1.0
            self.result, self.mp3, self.done_at = result, mp3, time.time()

    def fail(self, code: str, message: str) -> None:
        with self.lock:
            if self.status == "CANCELED":
                return
            self.status, self.error, self.done_at = "FAILED", {"code": code, "message": message}, time.time()

    def mark_canceled(self) -> None:
        with self.lock:
            if self.status in ("QUEUED", "RUNNING"):
                self.status, self.done_at = "CANCELED", time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return {"songId": self.id, "status": self.status, "step": self.step, "progress": round(self.progress, 2),
                    "result": self.result if self.status == "DONE" else None, "error": self.error}


class Store:
    def __init__(self, engine, log=print):
        self.engine = engine
        self.log = log
        self.sessions: dict[str, LiveSession] = {}
        self.jobs: dict[str, Job] = {}
        # 완성하기를 부른 세션 → 노래 번호. 앱이 응답을 못 받고 완성하기를 다시 불러도 같은 노래를 돌려준다
        self.session_songs: dict[str, str] = {}
        self.lock = threading.Lock()
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self._sweep_tmp(0)
        threading.Thread(target=self._cleanup_loop, name="cleanup", daemon=True).start()

    # ---------- 세션 ----------
    def create_session(self, members: list[Member]) -> LiveSession:
        sid = "s" + secrets.token_hex(4)
        s = self.engine.session_class(sid, members, self.engine, self.log)
        with self.lock:
            self.sessions[sid] = s
        return s

    def get_session(self, sid: str) -> LiveSession | None:
        with self.lock:
            return self.sessions.get(sid)

    def drop_session(self, sid: str) -> bool:
        with self.lock:
            s = self.sessions.pop(sid, None)
            jid = self.session_songs.get(sid)
        if jid is not None:
            # 완성하기를 부른 뒤 그만둔 경우(앱이 응답을 못 받음): 만들던 노래도 멈춘다.
            # 녹음은 노래 작업이 끝나면서 지운다 (작업 도중에 지우지 않게)
            self.delete_job(jid)
            return True
        if s is None:
            return False
        s.release()
        return True

    # ---------- 노래 ----------
    def job_for_session(self, sid: str) -> Job | None:
        """이미 완성하기를 부른 세션이면 그 노래 작업."""
        with self.lock:
            jid = self.session_songs.get(sid)
            return self.jobs.get(jid) if jid else None

    def start_job(self, session: LiveSession, drive: dict | None, beat: dict | None) -> Job:
        with self.lock:
            jid = self.session_songs.get(session.sid)
            if jid in self.jobs:
                return self.jobs[jid]
            job = Job("c" + secrets.token_hex(4))
            self.jobs[job.id] = job
            self.session_songs[session.sid] = job.id
        self.engine.song_executor.submit(self._run, job, session, drive, beat)
        return job

    def start_job_from_audio(self, audio16: np.ndarray, members: list[Member], drive: dict | None,
                             beat: dict | None) -> Job:
        session = self.engine.session_class("one-" + secrets.token_hex(3), members, self.engine, self.log)
        step = int(SR_IN * CHUNK_MS / 1000)
        for i in range(0, len(audio16), step):
            session.add_chunk(i // step, float_to_pcm16(audio16[i:i + step]))
        return self.start_job(session, drive, beat)

    def get_job(self, song_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(song_id)

    def delete_job(self, song_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(song_id)
            if job is None:
                return False
            if job.status in ("QUEUED", "RUNNING"):
                job.cancel.set()
                job.mark_canceled()
                return True
            self.jobs.pop(song_id, None)
        self._remove_file(job.mp3)
        return True

    def _run(self, job: Job, session: LiveSession, drive: dict | None, beat: dict | None) -> None:
        t0 = time.time()

        def progress(step: str, p: float) -> None:
            if job.cancel.is_set():
                raise Cancelled()
            if time.time() - t0 > JOB_TIMEOUT_SEC:
                raise JobTimeout()
            job.update(step, p)

        def log(msg: str) -> None:
            self.log(f"[노래 {job.id}] {msg}")

        try:
            progress("FIND_WORDS", 0.05)
            utterances = session.finalize(timeout=90)
            audio = session.audio()
            seconds = len(audio) / SR_IN
            speech = sum(u.duration for u in utterances)
            # 마이크가 무음인지 알 수 있게 소리 크기도 남긴다 (-90dB에 가까우면 마이크에서 소리가 안 들어온 것)
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
            level = lambda v: 20 * np.log10(max(v, 3.2e-5))  # -90dB 아래는 -90으로
            log(f"녹음 {seconds:.1f}초, 말소리 구간 {len(utterances)}개 ({speech:.1f}초), "
                f"소리 크기 평균 {level(rms):.0f}dB / 최대 {level(peak):.0f}dB")
            if seconds < MIN_RECORDING_SEC:
                job.fail("TOO_SHORT", "녹음이 3초보다 짧아요")
                return
            if self.engine.name != "mock" and speech < 0.5:
                job.fail("NO_SPEECH", "목소리가 들리지 않았어요")
                return
            progress("FIND_WORDS", 0.25)
            stereo, result = self.engine.make_song(audio, utterances, session.members, drive, beat, progress, log)
            progress("FINISH", 0.95)
            from songmaker.dsp import SR
            from songmaker.mix import save_mp3

            path = TMP_DIR / f"{job.id}.mp3"
            save_mp3(stereo, SR, path)
            result["audio"] = {"url": f"/v1/songs/{job.id}/audio", "format": "mp3", "expiresInSec": KEEP_SEC}
            if job.cancel.is_set():
                raise Cancelled()
            job.finish(result, path)
            log(f"완성: '{result['title']}' {result['durationMs'] / 1000:.1f}초 (만드는 데 {time.time() - t0:.1f}초)")
        except Cancelled:
            job.mark_canceled()
            self._remove_file(TMP_DIR / f"{job.id}.mp3")
            log("그만둠")
        except JobTimeout:
            job.fail("TIMEOUT", f"{JOB_TIMEOUT_SEC}초 안에 끝나지 않았어요")
            log("시간 초과")
        except Exception as e:
            job.fail("INTERNAL", "노래를 만들다가 문제가 생겼어요")
            log(f"[오류] {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            session.release()
            with self.lock:
                if self.sessions.get(session.sid) is session:
                    self.sessions.pop(session.sid, None)

    # ---------- 정리 ----------
    @staticmethod
    def _remove_file(path: Path | None) -> None:
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    def _sweep_tmp(self, max_age: float) -> None:
        now = time.time()
        with self.lock:
            live = {j.mp3 for j in self.jobs.values() if j.mp3 is not None}
        for f in TMP_DIR.glob("*.mp3"):
            try:
                if f not in live and now - f.stat().st_mtime > max_age:
                    f.unlink()
            except OSError:
                pass

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(15)
            now = time.time()
            stale_sessions, stale_jobs = [], []
            with self.lock:
                for sid, s in list(self.sessions.items()):
                    if not s.finished and now - s.last_activity > KEEP_SEC:
                        stale_sessions.append(sid)
                for jid, j in list(self.jobs.items()):
                    if j.done_at is not None and now - j.done_at > KEEP_SEC:
                        stale_jobs.append(jid)
            for sid in stale_sessions:
                self.drop_session(sid)
                self.log(f"[세션 {sid}] 10분 동안 조각이 없어서 지움")
            for jid in stale_jobs:
                with self.lock:
                    j = self.jobs.pop(jid, None)
                if j is not None:
                    self._remove_file(j.mp3)
            with self.lock:
                for sid, jid in list(self.session_songs.items()):
                    if jid not in self.jobs:
                        self.session_songs.pop(sid, None)
            self._sweep_tmp(KEEP_SEC)
