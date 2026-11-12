"""음성 인식 (faster-whisper, CPU). 단어마다 시작과 끝 시간을 받는다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .text import Word

# 아이 의성어를 알아듣도록 살짝 힌트를 준다. 명사는 넣지 않는다 (인식 성능을 부풀리지 않으려고).
HINT = "둥둥, 랄랄라, 빵빵, 쿵쿵, 짝짝, 부릉부릉, 칙칙폭폭."


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", models_dir: Path | None = None,
                 compute_type: str = "int8", threads: int = 0, hint: bool = True, beam_size: int = 5):
        from faster_whisper import WhisperModel

        root = str(models_dir) if models_dir else None
        self.name = model
        self.hint = hint
        self.beam_size = beam_size  # 1이면 빠르지만 조금 덜 정확하다 (느린 PC용)
        self.model = WhisperModel(model, device="cpu", compute_type=compute_type,
                                  download_root=root, cpu_threads=threads)

    def transcribe(self, audio16k: np.ndarray) -> list[Word]:
        segments, _info = self.model.transcribe(
            audio16k,
            language="ko",
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 150},
            condition_on_previous_text=False,
            initial_prompt=HINT if self.hint else None,
            temperature=0.0,
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                t = w.word.strip()
                if t:
                    words.append(Word(t, float(w.start), float(w.end), float(w.probability)))
        return words


class FixedTranscriber:
    """시험용: 미리 정한 단어와 시간을 그대로 돌려준다."""

    def __init__(self, table: dict[str, list[Word]]):
        self.table = table
        self.name = "fixed"
        self.current = ""

    def transcribe(self, audio16k: np.ndarray) -> list[Word]:
        return list(self.table.get(self.current, []))
