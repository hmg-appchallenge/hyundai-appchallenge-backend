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

ENROLLED_MIN_SIM = 0.45  # 등록한 목소리로 확실히 보려면 이 정도는 비슷해야 한다
ENROLLED_MARGIN = 0.03  # 둘째로 비슷한 사람보다 이만큼은 더 비슷해야 한다
# 같은 사람이라도 등록할 때와 녹음할 때 마이크·자리가 다르면 0.1~0.4까지 떨어진다 (에뮬레이터 마이크로 재 봄).
# 다른 사람은 대개 0.25 아래라서, 등록한 가족이 있으면 0.25 이상은 그 가족으로 보고 그보다 낮을 때만 다른 사람으로 본다
NEW_VOICE_MAX_SIM = 0.25
NEW_VOICE_MIN_SEC = 1.0  # 등록한 가족이 있을 때 등록 안 한 가족을 새로 세우려면 이만큼 긴 구간이어야 한다
ADAPT_MIN_SIM = 0.33  # 등록 목소리와 이만큼 닮은 1초 넘는 구간으로 이번 녹음에서의 그 사람 목소리를 익힌다
ADAPT_SESSION_SIM = 0.60  # 이미 익힌 이번 녹음 목소리와 이만큼 닮아도 익힌다
ADAPT_MAX_SEC = 4.0  # 이보다 긴 구간은 여러 사람이 이어 말한 게 섞이기 쉬워서 익히는 데 안 쓴다
# 이번 녹음에서 익힌 목소리는 같은 마이크라 남끼리도 0.4~0.5까지 닮게 나온다 (같은 사람은 0.55~0.9).
# 그래서 등록 목소리와 같은 잣대로 보려고 이만큼 빼고 비교한다 (0.52 -> 0.25, 0.72 -> 0.45)
SESSION_OFFSET = 0.27
CLUSTER_JOIN_SIM = 0.52  # 등록 안 한 목소리 묶음에 넣는 기준
CLUSTER_PREFER_MARGIN = 0.10  # 애매한 구간은 등록 안 한 목소리 묶음이 이만큼 더 닮을 때만 그쪽으로
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
        self.adapted: dict[int, np.ndarray] = {}  # 등록한 가족별로 이번 녹음에서 익힌 목소리 (합)
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

    def _score(self, m: Member, emb: np.ndarray) -> float:
        """등록 목소리와 이번 녹음에서 익힌 목소리(SESSION_OFFSET만큼 빼서) 중 더 닮은 쪽."""
        sim = float(emb @ m.embedding)
        acc = self.adapted.get(m.member_id)
        if acc is not None:
            sim = max(sim, float(emb @ (acc / (np.linalg.norm(acc) + 1e-9))) - SESSION_OFFSET)
        return sim

    def _enrolled_sims(self, emb: np.ndarray) -> list[tuple[float, int]]:
        return sorted(((self._score(m, emb), m.member_id) for m in self.enrolled), reverse=True)

    @staticmethod
    def _confident(sims: list[tuple[float, int]]) -> int | None:
        if not sims:
            return None
        best, mid = sims[0]
        second = sims[1][0] if len(sims) > 1 else -1.0
        return mid if best >= ENROLLED_MIN_SIM and best - second >= ENROLLED_MARGIN else None

    def _best_cluster(self, emb: np.ndarray) -> tuple[float, int] | None:
        scored = [(float(emb @ c["centroid"]), i) for i, c in enumerate(self.clusters)]
        return max(scored) if scored else None

    def _maybe_adapt(self, mid: int, emb: np.ndarray, duration: float) -> None:
        """그 가족으로 본 긴 구간이 등록 목소리(또는 이미 익힌 목소리)와 꽤 닮았으면 이번 녹음 목소리로 익힌다.
        차 안 마이크로 녹음하면 등록할 때와 소리가 달라서, 같은 사람인데 낮게 나오는 걸 녹음 안에서 바로잡는다."""
        if duration < NEW_VOICE_MIN_SEC or duration > ADAPT_MAX_SEC:
            return
        m = next((x for x in self.enrolled if x.member_id == mid), None)
        if m is None:
            return
        acc = self.adapted.get(mid)
        own = float(emb @ (acc / (np.linalg.norm(acc) + 1e-9))) if acc is not None else -1.0
        if float(emb @ m.embedding) >= ADAPT_MIN_SIM or own >= ADAPT_SESSION_SIM:
            self.adapted[mid] = emb.copy() if acc is None else acc + emb

    def identify(self, emb: np.ndarray | None, duration: float, final: bool = True) -> int | None:
        """누가 말했는지. final은 말소리 구간이 닫힌 것(믿을 만함), 아니면 말하는 중인 조각 일부.
        등록한 가족이 있으면 애매한 목소리는 가장 닮은 등록 가족으로 보고, 확실히 다른 목소리일 때만
        등록 안 한 가족을 세운다 (다영님 확인: 혼자 말했는데 등록 안 한 가족 카드가 번갈아 켜지던 것)."""
        if emb is None:
            return None
        sims = self._enrolled_sims(emb)
        mid = self._confident(sims)
        if mid is None and sims and sims[0][0] >= NEW_VOICE_MAX_SIM:
            c = self._best_cluster(emb)
            if c is not None and c[0] >= CLUSTER_JOIN_SIM and c[0] >= sims[0][0] + CLUSTER_PREFER_MARGIN:
                return self._join(c[1], emb, duration) if final else self.clusters[c[1]]["member_id"]
            mid = sims[0][1]
        if mid is not None:
            if final:
                self._maybe_adapt(mid, emb, duration)
            return mid
        free = self.unregistered
        if not free:
            self.unknown_flag = True
            return None
        c = self._best_cluster(emb)
        if c is not None and c[0] >= CLUSTER_JOIN_SIM:
            # 아무도 등록 안 했으면 예전처럼 말하는 중인 조각으로도 묶음을 다듬는다
            return self._join(c[1], emb, duration) if final or not self.enrolled else self.clusters[c[1]]["member_id"]
        # 새 사람: 등록한 가족이 있으면 닫힌 긴 구간으로만 (말하는 중인 짧은 조각은 흔들려서)
        new_ok = (final and duration >= NEW_VOICE_MIN_SEC) if self.enrolled else duration >= 0.8
        if new_ok and len(self.clusters) < len(free):
            member = free[len(self.clusters)]
            self.clusters.append({"centroid": emb.copy(), "count": 1, "member_id": member.member_id})
            return member.member_id
        if c is not None and (c[0] >= NEW_VOICE_MAX_SIM or not self.enrolled):
            return self.clusters[c[1]]["member_id"]
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
        1) 등록 목소리와 꽤 닮은 긴 구간으로 이번 녹음에서의 가족 목소리를 다시 익힌다
        2) 확실히 닮았거나 애매한 구간은 가장 닮은 등록 가족, 등록한 누구와도 안 닮은 구간만
           비슷한 것끼리 묶어 등록 안 한 가족에게 처음 말한 차례로 붙인다 (1초도 안 되는 묶음은 누군지 모름)"""
        utts = [u for u in self.utterances if u.embedding is not None]
        if not utts:
            return
        enrolled = self.enrolled
        self.adapted = {}
        for u in sorted(utts, key=lambda x: -x.duration):
            raw = sorted(((float(u.embedding @ m.embedding), m.member_id) for m in enrolled), reverse=True)
            mid = self._confident(raw) if raw and raw[0][0] >= ENROLLED_MIN_SIM else None
            if mid is None and raw and raw[0][0] >= ADAPT_MIN_SIM and (len(raw) == 1 or raw[0][0] - raw[1][0] >= ENROLLED_MARGIN):
                mid = raw[0][1]
            if mid is not None:
                self._maybe_adapt(mid, u.embedding, u.duration)
        gray: list[tuple[Utterance, int, float]] = []
        rest: list[Utterance] = []
        for u in utts:
            sims = self._enrolled_sims(u.embedding)
            mid = self._confident(sims)
            if mid is not None:
                u.member_id = mid
            elif sims and sims[0][0] >= NEW_VOICE_MAX_SIM:
                gray.append((u, sims[0][1], sims[0][0]))
                u.member_id = sims[0][1]
            else:
                rest.append(u)
        free = self.unregistered
        if not free:
            for u in rest:
                u.member_id = None
            return
        if not rest:
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
        if enrolled:
            # 등록한 가족이 있으면 1초도 안 되는 다른 목소리(기침, 잡음, 남의 짧은 말)는 누군지 모름으로 둔다
            for g in [g for g in groups if sum(u.duration for u in g) < NEW_VOICE_MIN_SEC]:
                for u in g:
                    u.member_id = None
                groups.remove(g)
        groups.sort(key=lambda g: min(u.start for u in g))
        cents = []
        for k, g in enumerate(groups[:len(free)]):
            member = free[k].member_id
            for u in g:
                u.member_id = member
            cents.append((centroid(g), member))
        # 애매해서 등록 가족으로 둔 구간이 등록 안 한 목소리 묶음과 훨씬 닮았으면 그쪽으로
        for u, _mid, sim in gray:
            scored = [(float(u.embedding @ c), m) for c, m in cents]
            if scored:
                cs, cm = max(scored)
                if cs >= CLUSTER_JOIN_SIM and cs >= sim + CLUSTER_PREFER_MARGIN:
                    u.member_id = cm

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
                mid = self.identify(self.engine.encoder.embed(part), len(part) / SR_IN, final=False)
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
