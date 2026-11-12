"""모델 내려받기: 화자 모델·VAD(sherpa-onnx, GitHub) + 음성 인식·TTS·형태소 분석기."""
from __future__ import annotations

import sys
import time
import urllib.request

from .config import AI_DIR, MODELS_DIR, SHERPA_DIR, SHERPA_URLS


def fetch(url: str, dest, log) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    tmp.replace(dest)
    log(f"  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}초)")


def download_all(stt_model: str, log) -> int:
    SHERPA_DIR.mkdir(parents=True, exist_ok=True)
    log("화자 구분·말소리 구간 모델 확인 중...")
    for name, url in SHERPA_URLS.items():
        dest = SHERPA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            log(f"  {name} 있음")
            continue
        try:
            fetch(url, dest, log)
        except Exception as e:
            log(f"  [실패] {name}: {type(e).__name__}: {e}")
            return 1
    if str(AI_DIR) not in sys.path:
        sys.path.insert(0, str(AI_DIR))
    from songmaker.download import download

    download(MODELS_DIR, stt_model, log)
    log("모델 준비 끝.")
    return 0
