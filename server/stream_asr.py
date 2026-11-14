"""말하는 동안 바로 받아쓰기 (sherpa-onnx 스트리밍 zipformer, 한국어 KsponSpeech, Apache-2.0).

Whisper는 정확하지만 허깅페이스 CPU에서 한 번 듣는 데 몇 초가 걸려서, 말소리 구간이 닫힌 뒤에야 단어가 나온다.
이 모델은 2초 조각마다 0.1~0.2초 만에 이어서 받아써서, 말한 그 조각 응답에 단어를 실을 수 있다.
덜 정확해서 화면의 '지금까지 들린 단어'에만 쓰고, 노래 가사에는 Whisper가 다시 들은 단어를 쓴다.
모델 파일이 없거나 불러오지 못하면 꺼진다 (그러면 예전처럼 구간이 닫힌 뒤에 단어가 간다).
"""
from __future__ import annotations

import threading

import numpy as np

from .config import SR_IN, STREAM_DIR, STREAM_FILES


class StreamAsr:
    def __init__(self, models_dir=STREAM_DIR, log=print):
        self._rec = None
        self._lock = threading.Lock()  # 세션마다 스트림은 따로지만, 계산은 하나씩 (조각 하나에 0.1~0.2초)
        paths = {name: models_dir / name for name in STREAM_FILES}
        try:
            if all(p.exists() for p in paths.values()):
                import sherpa_onnx

                self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(paths["tokens.txt"]),
                    encoder=str(paths["encoder-epoch-99-avg-1.int8.onnx"]),
                    decoder=str(paths["decoder-epoch-99-avg-1.int8.onnx"]),
                    joiner=str(paths["joiner-epoch-99-avg-1.int8.onnx"]),
                    num_threads=1,
                    sample_rate=SR_IN,
                    feature_dim=80,
                    decoding_method="greedy_search",  # 한 번 낸 글자는 바뀌지 않는다
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=1.2,  # 아무 말 없이 1.2초
                    rule2_min_trailing_silence=0.6,  # 말한 뒤 0.6초 쉬면 한 마디가 끝난 것
                    rule3_min_utterance_length=20,  # 쉬지 않아도 20초마다 끊는다
                )
        except Exception as e:
            log(f"[실시간 받아쓰기 꺼짐] {type(e).__name__}: {e}")
            self._rec = None
        self.name = "zipformer-ko-streaming" if self._rec is not None else "off"

    @property
    def ready(self) -> bool:
        return self._rec is not None

    def new_stream(self):
        return self._rec.create_stream() if self._rec is not None else None

    def feed(self, stream, x: np.ndarray) -> tuple[str, bool]:
        """조각을 넣고 (지금 마디에서 받아쓴 글, 마디가 끝났는지)를 돌려준다. 끝났으면 다음 마디를 위해 비운다."""
        with self._lock:
            stream.accept_waveform(SR_IN, np.ascontiguousarray(x, dtype=np.float32))
            while self._rec.is_ready(stream):
                self._rec.decode_stream(stream)
            text = self._rec.get_result(stream)
            text = text if isinstance(text, str) else getattr(text, "text", "")
            end = bool(self._rec.is_endpoint(stream))
            if end:
                self._rec.reset(stream)
        return text, end
