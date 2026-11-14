"""녹음 세션: 2초 조각을 받는 동안 말소리 구간을 찾아 누가 말했는지와 새 단어를 알아낸다.

단어는 두 길로 찾는다. 조각이 올 때마다 실시간 받아쓰기로 바로 찾아 그 조각 응답에 싣고(화면용, 덜 정확),
말이 끊겨 구간이 닫히면 Whisper로 구간을 다시 듣는다(정확, 노래 가사에 씀). 같은 단어는 한 번만 보낸다.
완성할 때는 모아 둔 구간(시간, 화자, 단어)을 그대로 노래 만들기에 넘긴다.
녹음 원본은 세션이 끝나면(완성·취소·10분 동안 조각 없음) 지운다.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from .audio_io import pcm16_to_float
from .config import MAX_RECORDING_SEC, SR_IN
from .speaker import Segment, voiced_part

ENROLLED_MIN_SIM = 0.45  # 등록한 목소리로 보려면 이 정도는 비슷해야 한다
ENROLLED_MARGIN = 0.03  # 둘째로 비슷한 사람보다 이만큼은 더 비슷해야 한다
CLUSTER_JOIN_SIM = 0.52  # 등록 안 한 목소리 묶음에 넣는 기준
LOOSE_SIM = 0.30  # 모두 등록한 경우, 이보다 낮으면 모르는 사람
CLUSTER_MERGE_SIM = 0.50  # 완성할 때 등록 안 한 목소리 구간끼리 합치는 기준 (평균 유사도)
GAP_SKIP_PENDING = 3  # 빠진 조각 뒤로 이만큼 쌓일 때까지 안 오면 잃어버린 것으로 보고 건너뛴다 (2초 조각이면 6초)
STT_BATCH_SEC = 25.0  # 한 번에 인식할 말소리 길이 (Whisper 창 30초 안)
STT_GAP_SEC = 0.4  # 구간 사이에 넣는 빈소리
PRE_ROLL_SEC = 0.3  # VAD는 첫소리(토끼의 ㅌ, 아빠랑의 아)를 늦게 잡을 때가 있어서 Whisper에는 구간 앞을 조금 더 들려준다
POST_ROLL_SEC = 0.15  # 끝소리도 조금 (바로 이어 말하는 중이면 안 붙인다)


def word_key(text: str, kind: str) -> str:
    """같은 단어인지 볼 때 쓰는 열쇠. 같은 소리를 몇 번 반복했는지만 다른 의성어(둥둥, 둥둥둥둥)는 한 번만 보낸다."""
    if kind == "onomatopoeia":
        for u in (1, 2):
            unit = text[:u]
            if len(text) >= 2 * u and len(text) % u == 0 and unit * (len(text) // u) == text:
                return "ono:" + unit
    return text


@dataclass
class Member:
    member_id: int
    instrument: str
    embedding: np.ndarray | None = None


@dataclass
class Utterance:
    start: float
    end: float
    audio: np.ndarray
    embedding: np.ndarray | None
    member_id: int | None
    words: list = field(default_factory=list)  # songmaker.text.Word (녹음 전체 기준 시간)
    analyzed: bool = False
    audio_start: float | None = None  # audio가 녹음에서 시작하는 시간 (앞뒤를 붙여서 start와 다를 수 있다)

    @property
    def duration(self) -> float:
        return self.end - self.start


class LiveSession:
    def __init__(self, sid: str, members: list[Member], engine, log=print):
        self.sid = sid
        self.members = members
        self.engine = engine
        self.log = log
        self.lock = threading.RLock()
        self.chunks: list[np.ndarray] = []
        self.samples = 0
        self.next_seq = 0
        self.pending: dict[int, bytes] = {}
        self.vad = engine.new_vad()
        self.utterances: list[Utterance] = []
        self.futures: list[Future] = []
        # 음성 인식은 말소리 구간을 모아서 한 번에 한다 (Whisper는 짧은 말도 30초 창 하나를 통째로 계산해서)
        self.stt_pending: list[Utterance] = []
        self.stt_running = False
        self.heard_until = 0  # Whisper에 넘긴 소리의 끝 (샘플). 다음 구간 앞을 붙일 때 겹치지 않게
        self.word_queue: list[dict] = []
        self.seen_words: set[str] = set()
        asr = getattr(engine, "stream_asr", None)
        self.stream = asr.new_stream() if asr is not None and asr.ready else None  # 실시간 받아쓰기 (없으면 None)
        self.stream_text = ""  # 지난 조각까지 받아쓴 글 (지금 마디)
        self.clusters: list[dict] = []  # 등록 안 한 목소리 묶음: centroid, count, member_id
        self.finished = False
        self.last_activity = time.time()
        self.created = time.time()
        self.unknown_flag = False

    # ---------- 화자 ----------
    @property
    def enrolled(self) -> list[Member]:
        return [m for m in self.members if m.embedding is not None]

    @property
    def unregistered(self) -> list[Member]:
        return [m for m in self.members if m.embedding is None]

    def _enrolled_sims(self, emb: np.ndarray) -> list[tuple[float, int]]:
        return sorted(((float(emb @ m.embedding), m.member_id) for m in self.enrolled), reverse=True)

    @staticmethod
    def _confident(sims: list[tuple[float, int]]) -> int | None:
        if not sims:
            return None
        best, mid = sims[0]
        second = sims[1][0] if len(sims) > 1 else -1.0
        return mid if best >= ENROLLED_MIN_SIM and best - second >= ENROLLED_MARGIN else None

    def identify(self, emb: np.ndarray | None, duration: float) -> int | None:
        """녹음 중 실시간 판단. 짧은 소리로는 새 묶음을 만들지 않는다 (완성할 때 다시 정리한다)."""
        if emb is None:
            return None
        sims = self._enrolled_sims(emb)
        mid = self._confident(sims)
        if mid is not None:
            return mid
        free = self.unregistered
        if free:
            scored = sorted(((float(emb @ c["centroid"]), i) for i, c in enumerate(self.clusters)), reverse=True)
            if scored and scored[0][0] >= CLUSTER_JOIN_SIM:
                return self._join(scored[0][1], emb, duration)
            if duration >= 0.8 and len(self.clusters) < len(free):
                member = free[len(self.clusters)]
                self.clusters.append({"centroid": emb.copy(), "count": 1, "member_id": member.member_id})
                return member.member_id
            if scored:
                return self.clusters[scored[0][1]]["member_id"]
        if sims and sims[0][0] >= LOOSE_SIM:
            return sims[0][1]
        self.unknown_flag = True
        return None

    def _join(self, idx: int, emb: np.ndarray, duration: float) -> int:
        c = self.clusters[idx]
        if duration >= 0.6:
            total = c["centroid"] * c["count"] + emb
            c["count"] += 1
            c["centroid"] = total / (np.linalg.norm(total) + 1e-9)
        return c["member_id"]

    def refine_speakers(self) -> None:
        """완성할 때 모든 구간을 한꺼번에 보고 화자를 다시 정한다.
        등록한 목소리와 확실히 닮은 구간은 그 사람, 나머지는 비슷한 것끼리 묶어 등록 안 한 가족에게 차례로 붙인다."""
        utts = [u for u in self.utterances if u.embedding is not None]
        rest = []
        for u in utts:
            mid = self._confident(self._enrolled_sims(u.embedding))
            if mid is not None:
                u.member_id = mid
            else:
                rest.append(u)
        free = self.unregistered
        if not rest:
            return
        if not free:
            for u in rest:
                sims = self._enrolled_sims(u.embedding)
                u.member_id = sims[0][1] if sims and sims[0][0] >= LOOSE_SIM else None
            return

        def centroid(group: list[Utterance]) -> np.ndarray:
            v = sum(u.embedding * max(u.duration, 0.2) for u in group)
            return v / (np.linalg.norm(v) + 1e-9)

        def link(a: list[Utterance], b: list[Utterance]) -> float:
            # 묶음끼리 구간 하나하나의 평균 유사도 (묶음이 커져도 기준이 느슨해지지 않게)
            return float(np.mean([x.embedding @ y.embedding for x in a for y in b]))

        groups = [[u] for u in rest]
        while len(groups) > 1:
            best, pair = -2.0, (0, 1)
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    sim = link(groups[i], groups[j])
                    if sim > best:
                        best, pair = sim, (i, j)
            if best < CLUSTER_MERGE_SIM and len(groups) <= len(free):
                break
            i, j = pair
            groups[i] += groups.pop(j)
        groups.sort(key=lambda g: min(u.start for u in g))
        for k, g in enumerate(groups):
            member = free[k].member_id
            c = centroid(g)
            for u in g:
                sims = self._enrolled_sims(u.embedding)
                own = float(u.embedding @ c) if len(g) > 1 else 0.0
                u.member_id = sims[0][1] if sims and sims[0][0] > max(own + 0.05, LOOSE_SIM) else member

    # ---------- 조각 ----------
    @property
    def received_sec(self) -> float:
        return self.samples / SR_IN

    def add_chunk(self, seq: int, data: bytes) -> dict:
        with self.lock:
            self.last_activity = time.time()
            speaking: set[int] = set()
            if seq >= self.next_seq and seq not in self.pending and len(self.pending) < 16:
                self.pending[seq] = data
            speaking |= self._drain()
            words, self.word_queue = self.word_queue, []
            unknown, self.unknown_flag = self.unknown_flag, False
            return {"seq": seq, "receivedSec": round(self.received_sec, 2), "speaking": sorted(speaking),
                    "unregisteredVoice": unknown, "newWords": words}

    def _drain(self, force: bool = False) -> set[int]:
        """받은 조각을 순서대로 처리한다. 앱이 재시도 끝에 포기한 조각이 있으면 기다리다가 건너뛴다.
        force=True(완성할 때)면 빠진 조각을 기다리지 않고 남은 것을 모두 처리한다."""
        speaking: set[int] = set()
        while self.pending:
            if self.next_seq not in self.pending:
                if not force and len(self.pending) < GAP_SKIP_PENDING:
                    break
                last = min(self.pending) - 1
                missing = f"{self.next_seq}" if last == self.next_seq else f"{self.next_seq}~{last}"
                self.log(f"[{self.sid}] 조각 {missing}번이 오지 않아 건너뛰어요")
                self.next_seq = min(self.pending)
            raw = self.pending.pop(self.next_seq)
            self.next_seq += 1
            speaking |= self._process(pcm16_to_float(raw))
        return speaking

    def would_exceed(self, nbytes: int) -> bool:
        return (self.samples + nbytes // 2) / SR_IN > MAX_RECORDING_SEC + 1.0

    def _process(self, x: np.ndarray) -> set[int]:
        if x.size == 0:
            return set()
        chunk_start = self.samples
        self.chunks.append(x)
        self.samples += len(x)
        speaking: set[int] = set()
        latest = None  # 이 조각에서 마지막으로 알아본 사람 (실시간 단어에 붙인다)
        for seg in self.vad.accept(x):
            mid = self._add_utterance(seg)
            if mid is not None and seg.start + len(seg.samples) > chunk_start:
                speaking.add(mid)
                latest = mid
        if self.vad.speech_now():
            part = voiced_part(x)
            if len(part) >= int(0.8 * SR_IN):
                mid = self.identify(self.engine.encoder.embed(part), len(part) / SR_IN)
                if mid is not None:
                    speaking.add(mid)
                    latest = mid
        if self.stream is not None:
            self._stream_words(x, latest)
        return speaking

    def _stream_words(self, x: np.ndarray, member_id: int | None) -> None:
        """실시간 받아쓰기에 조각을 넣고, 새로 들린 단어를 이번 응답에 싣는다.
        글 맨 끝 단어는 다음 조각에서 더 이어질 수 있어서(강아 -> 강아지) 마디가 끝나거나 글이 그대로일 때까지 미룬다."""
        try:
            text, end = self.engine.stream_asr.feed(self.stream, x)
            settled = end or text == self.stream_text
            self.stream_text = "" if end else text
            tail = text.rstrip()
            if not tail:
                return
            found = self.engine.stream_notable(text, member_id)
            if not settled:
                found = [w for w in found if not tail.endswith(w["text"])]
            self._queue_words(found)
        except Exception as e:  # 받아쓰기가 탈이 나도 녹음은 계속 (구간이 닫히면 Whisper가 듣는다)
            self.log(f"[세션 {self.sid}] 실시간 받아쓰기 멈춤: {type(e).__name__}: {e}")
            self.stream = None

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        """녹음에서 lo~hi 샘플 (받은 조각 끝에서부터 거슬러 찾는다)."""
        parts, pos = [], self.samples
        for c in reversed(self.chunks):
            if pos <= lo:
                break
            parts.append(c)
            pos -= len(c)
        x = np.concatenate(parts[::-1]) if parts else np.zeros(0, np.float32)
        return x[max(0, lo - pos):max(0, hi - pos)]

    def _add_utterance(self, seg: Segment) -> int | None:
        emb = self.engine.encoder.embed(seg.samples)
        mid = self.identify(emb, len(seg.samples) / SR_IN)
        seg_end = seg.start + len(seg.samples)
        lo = min(seg.start, max(0, self.heard_until, seg.start - int(PRE_ROLL_SEC * SR_IN)))
        hi = seg_end if self.vad.speech_now() else max(seg_end, min(self.samples, seg_end + int(POST_ROLL_SEC * SR_IN)))
        audio = self._slice(lo, hi) if (lo, hi) != (seg.start, seg_end) else seg.samples
        if len(audio) != hi - lo:  # 녹음에서 못 찾으면 VAD가 준 소리 그대로
            audio, lo, hi = seg.samples, seg.start, seg_end
        self.heard_until = max(self.heard_until, hi)
        u = Utterance(seg.start_sec, seg.end_sec, audio, emb, mid, audio_start=lo / SR_IN)
        self.utterances.append(u)
        self.stt_pending.append(u)
        if not self.stt_running:
            self.stt_running = True
            self.futures.append(self.engine.live_executor.submit(self._stt_loop))
        return mid

    def _take_batch(self) -> list[Utterance]:
        """기다리는 말소리 구간을 STT_BATCH_SEC 안에서 앞에서부터 꺼낸다. 없으면 인식 일꾼을 멈춘다."""
        with self.lock:
            batch, total = [], 0.0
            while self.stt_pending:
                u = self.stt_pending[0]
                d = len(u.audio) / SR_IN + STT_GAP_SEC
                if batch and total + d > STT_BATCH_SEC:
                    break
                batch.append(self.stt_pending.pop(0))
                total += d
            if not batch:
                self.stt_running = False
            return batch

    def _stt_loop(self) -> None:
        """인식하는 동안 새로 쌓인 구간은 다음 묶음으로 한꺼번에 인식한다 (녹음이 길수록 묶음이 커진다)."""
        while True:
            batch = self._take_batch()
            if not batch:
                return
            try:
                self._analyze_batch(batch)
            except Exception as e:  # 인식이 실패해도 세션은 계속
                self.log(f"[세션 {self.sid}] 말소리 인식 실패: {type(e).__name__}: {e}")
            finally:
                for u in batch:
                    u.analyzed = True

    def _analyze_batch(self, batch: list[Utterance]) -> None:
        gap = np.zeros(int(STT_GAP_SEC * SR_IN), np.float32)
        parts, spans, pos = [], [], 0.0
        for u in batch:
            parts += [gap, u.audio]
            pos += STT_GAP_SEC
            spans.append((pos, pos + len(u.audio) / SR_IN, u))
            pos += len(u.audio) / SR_IN
        words = self.engine.transcribe_utterance(np.concatenate(parts), 0.0)
        per: dict[int, list] = {id(u): [] for u in batch}
        for w in words:
            mid_t = (w.start + w.end) / 2
            # 단어 가운데가 들어 있는 구간, 사이 빈칸이면 가장 가까운 구간
            start, end, u = min(spans, key=lambda s: 0.0 if s[0] <= mid_t <= s[1] else min(abs(mid_t - s[0]), abs(mid_t - s[1])))
            shift = (u.audio_start if u.audio_start is not None else u.start) - start
            per[id(u)].append(type(w)(w.text, max(u.start, w.start + shift), max(u.start, w.end + shift), w.prob))
        found_all = []
        for _start, _end, u in spans:
            u.words = per[id(u)]
            found_all += self.engine.notable_words(u.words, u.member_id)
        self._queue_words(found_all)

    def _queue_words(self, found: list[dict]) -> None:
        """처음 들은 단어만 다음 조각 응답에 실어 보낸다."""
        with self.lock:
            for w in found:
                key = word_key(w["text"], w["kind"])
                if key not in self.seen_words:
                    self.seen_words.add(key)
                    self.word_queue.append(w)

    # ---------- 마무리 ----------
    def audio(self) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)

    def finalize(self, timeout: float = 90.0) -> list[Utterance]:
        """남은 소리를 구간으로 닫고, 인식이 다 끝날 때까지 기다린다."""
        with self.lock:
            self.finished = True
            self._drain(force=True)
            for seg in self.vad.flush():
                self._add_utterance(seg)
            futures = list(self.futures)
        deadline = time.time() + timeout
        for f in futures:
            try:
                f.result(timeout=max(0.1, deadline - time.time()))
            except Exception:
                pass
        with self.lock:
            self.refine_speakers()
            return list(self.utterances)

    def release(self) -> None:
        """녹음 원본을 지운다."""
        with self.lock:
            self.chunks = []
            self.pending = {}
            for u in self.utterances:
                u.audio = np.zeros(0, np.float32)
