"""녹음 세션: 2초 조각을 받는 동안 말소리 구간을 찾아 누가 말했는지와 새 단어를 알아낸다.

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
        self.word_queue: list[dict] = []
        self.seen_words: set[str] = set()
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
        for seg in self.vad.accept(x):
            mid = self._add_utterance(seg)
            if mid is not None and seg.start + len(seg.samples) > chunk_start:
                speaking.add(mid)
        if self.vad.speech_now():
            part = voiced_part(x)
            if len(part) >= int(0.8 * SR_IN):
                mid = self.identify(self.engine.encoder.embed(part), len(part) / SR_IN)
                if mid is not None:
                    speaking.add(mid)
        return speaking

    def _add_utterance(self, seg: Segment) -> int | None:
        emb = self.engine.encoder.embed(seg.samples)
        mid = self.identify(emb, len(seg.samples) / SR_IN)
        u = Utterance(seg.start_sec, seg.end_sec, seg.samples, emb, mid)
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
                d = u.duration + STT_GAP_SEC
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
            shift = u.start - start
            per[id(u)].append(type(w)(w.text, max(u.start, w.start + shift), max(u.start, w.end + shift), w.prob))
        found_all = []
        for _start, _end, u in spans:
            u.words = per[id(u)]
            found_all += self.engine.notable_words(u.words, u.member_id)
        with self.lock:
            for w in found_all:
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
