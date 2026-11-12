# AI 노래 시제품 (청취 테스트용)

> 서버가 가져다 쓰는 노래 엔진(`songmaker`)을 혼자 돌려 보는 방법이에요. 서버 실행은 맨 위 [README](../README.md)를 보세요.

녹음 파일 → 음성 인식 → 명사·의성어 고르기 → 가사(서술 틀) → 보컬 → 주행 비트 합성 → mp3.
전부 이 PC의 CPU에서 돌아가고, 따로 받을 API 키는 없어.

## 처음 한 번: `setup-ai.bat`

`kidsrhythm-dev` 폴더의 `setup-ai.bat`을 더블클릭해.

- 파이썬 3.12로 가상환경(`ai\.venv`)을 만들고 패키지를 깔아.
- 모델을 받아: 음성 인식 faster-whisper large-v3-turbo(약 1.6GB), TTS Supertonic-3.
- 전부 합쳐 3~4GB쯤 들어. 중간에 빨간 글씨가 나와도 마지막에 "AI 환경 준비 완료!"가 뜨면 괜찮아.

## 노래 만들기: `make-song.bat`

- 녹음 파일(최대 3개)을 `make-song.bat` 위에 끌어다 놓거나, `ai\input` 폴더에 넣고 더블클릭.
- 녹음이 하나도 없으면 시험용 가짜 목소리로 만들어. 이때 TTS 목소리 10개 미리 듣기 파일(`voices_preview.mp3`)도 같이 나와.
- 결과는 `ai\output\날짜-시간` 폴더에 생기고, 끝나면 폴더가 열려.

## 결과 파일

| 파일 | 내용 |
|---|---|
| `city_A_child.mp3` | 시내 주행(정차 잦음, 셔플) + 아이 목소리만 |
| `city_B_tts.mp3` | 시내 주행 + TTS만 |
| `city_C_mix.mp3` | 시내 주행 + TTS 가사와 아이 목소리 섞기 |
| `highway_*.mp3` | 고속도로(일정한 고속, 4비트) 세 가지 |
| `summary.txt` | 알아들은 말, 가사, 파일 목록 |
| `*_response.json` | 앱이 받게 될 응답과 같은 모양의 결과 |

## 비트 (Claude가 작곡)

Claude API가 주행 정보(속도, 정차, 가속·감속 위치), 장소 분위기, 계절, 아이 말 빠르기를 보고 비트 악보를 JSON으로 써. BPM, 스윙, 조, 16칸 드럼 패턴, 화음, 필인, 가족 타악기 패턴, 후렴 멜로디가 들어가고, 이 PC의 엔진이 그 악보대로 연주해.

- `ai-keys.env`에 `ANTHROPIC_API_KEY=...` 한 줄을 넣으면 켜져. 키가 없거나 호출이 실패하면 기획 문서의 규칙 비트로 만들어.
- Claude에는 아이가 한 말의 내용(명사)과 정확한 위치는 보내지 않아. 말 빠르기, 의성어 종류, 장소 분위기(도심, 고속도로 같은 말)만 보내.
- 결과 폴더의 `*_beat.json`이 Claude가 쓴 악보이고, `summary.txt`에 왜 그렇게 정했는지가 적혀.
- 규칙 비트와 비교하려면 `--beat both`.

## 녹음 규칙

- 파일 하나가 가족 한 명이고, 순서대로 북, 탬버린, 셰이커가 붙어.
- 단어 사이를 반 초쯤 띄우면 잘라 내기가 깔끔해.
- 가사에는 아이가 말한 명사 중 2음절 이상, 최대 4개가 들어가.

## 고칠 수 있는 설정 (명령줄)

`ai` 폴더에서 `.venv\Scripts\python.exe -m songmaker --help`

- `--voice F3`: TTS 목소리 바꾸기 (F1~F5, M1~M5)
- `--tts-pitch 4`: TTS 음높이 (반음, 기본 3)
- `--scenarios city,highway,suburb`: 주행 상황
- `--season spring`: 계절 악기 강제로 정하기 (기본은 오늘 날짜)
- `--stt-model small`: 빠르지만 덜 정확한 음성 인식
- `--no-hint`: 의성어 힌트 없이 인식 (아이 음성 인식 정확도 검증용)
- `--beat rule|claude|both`: 비트 작곡 방식 (기본 auto: 키 있으면 Claude)
- `--place 바닷가`, `--temp 28`: 장소 분위기, 바깥 온도
- `--claude-model claude-sonnet-5-5`: 비트 작곡 모델 바꾸기 (기본 claude-haiku-5-5)

## 기록

실행 기록은 `ai\logs`에 남아. 문제가 생기면 그 기록을 확인해.
