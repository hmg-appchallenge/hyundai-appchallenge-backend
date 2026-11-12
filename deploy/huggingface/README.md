---
title: Kidsrhythm Server
emoji: 🎵
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 차량 리듬 동요 노래 생성 서버

현대자동차그룹 차량용 앱 개발 공모전 출품작 "차량 리듬 동요"(Pleos Connect Kids 앱)가 쓰는 노래 생성 서버예요.
앱이 보낸 녹음에서 아이가 한 말을 알아듣고, 그 말로 가사를 쓰고(Claude), 주행 속도에 맞는 동요를 만들어(ElevenLabs Music) 돌려줘요.

- API 명세: [`docs/API.md`](docs/API.md), 서버 주소 뒤에 `/docs`를 붙이면 Swagger 화면이 나와요
- 서버 상태: 주소 뒤에 `/v1/health`
- `/v1/` 아래 요청은 앱 전용 팀 키(`X-Api-Key`)가 있어야 해요
- 녹음 소리는 바깥 AI로 보내지 않고, 받아쓴 글자만 가사를 쓰는 데 보내요. 녹음과 목소리 특징은 서버에 저장하지 않아요

## 구성

| 폴더 | 내용 |
|---|---|
| `server/` | HTTP API (FastAPI): 녹음 세션, 목소리 특징, 노래 작업, 외부 AI 호출 |
| `ai/songmaker/` | 음성 인식(faster-whisper), 형태소 분석(Kiwi), 반주 합성, 노래 조립 |
| `docs/API.md` | 앱과 서버가 주고받는 형식 |
