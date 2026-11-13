# 차량 리듬 동요 노래 생성 서버

Pleos Connect Kids 앱 "차량 리듬 동요"의 노래 생성 서버(AI·백엔드)예요. 앱(프론트엔드)은 [hyundai-appchallenge-frontend](https://github.com/hmg-appchallenge/hyundai-appchallenge-frontend)에 있고, 둘이 주고받는 형식은 [docs/API.md](docs/API.md)예요.

서버 PC에서는 이 저장소를 받으면 돼요.

```bat
git clone https://github.com/hmg-appchallenge/hyundai-appchallenge-backend.git kidsrhythm-server
```

## 하는 일

1. 녹음하는 동안 앱이 2초 조각을 보내면, 누가 소리 냈는지(화자 구분)와 새로 들린 단어를 바로 돌려줘요.
2. 완성하기를 누르면 녹음 전체를 받아쓰고(faster-whisper), 명사와 의성어를 골라요(Kiwi).
3. 가사와 제목은 Claude가 가족이 실제로 한 말로 쓰고, 반주와 보컬은 ElevenLabs Music이 주행 속도에 맞는 빠르기로 만들어요. 아이 목소리 조각도 박자에 맞춰 얹어요.
4. 앱이 mp3와 가사(줄마다 시간, 강조 위치)를 받아 가면 서버는 지워요.

- 바깥 AI에는 받아쓴 글자만 보내요. 녹음 소리와 목소리 특징은 보내지 않고, 서버에 남기지도 않아요 (docs/API.md "기본 원칙").
- 바깥 생성이 실패하면 서버 안 엔진(서술 틀 가사, 합성 반주)으로 대신 만들어요.

## 폴더

| 위치 | 내용 |
|---|---|
| `server/` | FastAPI 서버. `app.py` 엔드포인트, `live.py` 녹음 세션, `jobs.py` 노래 작업, `remote_song.py` Claude·ElevenLabs, `speaker.py` 화자 구분, `engines.py` 모델, `smoketest.py` 전체 흐름 시험, `tools.py` 진단·배포 도구 |
| `ai/songmaker/` | 받아쓰기·가사·합성 엔진. 서버가 가져다 쓰고, 혼자 돌려 볼 수도 있어요 ([ai/README.md](ai/README.md)) |
| `docs/API.md` | 앱과 주고받는 형식. 프론트 저장소의 `docs/API.md`와 같은 내용이라 바꿀 때 같이 고쳐요 |
| `deploy/huggingface/` | Hugging Face Space(Docker)에 올릴 때 쓴 Dockerfile과 Space 설명 |
| `*.bat`, `*.ps1` | Windows에서 환경 만들기, 서버 켜기, 바깥에 열기 |

## 처음 준비 (Windows)

1. Python 3.12(또는 3.11)를 설치해요.
2. `setup-server.bat`을 더블클릭해요. 가상환경(`ai\.venv`), 패키지, 모델을 준비해요 (합쳐서 4GB 안팎, 처음엔 몇 분 걸려요).
3. `ai-keys.env.example`을 `ai-keys.env`로 복사하고 키를 채워요. 키는 팀에서 따로 받아요.

`setup-server.bat` 대신 직접 할 때:

```bat
setup-ai.bat
ai\.venv\Scripts\python -m pip install -r server\requirements.txt
ai\.venv\Scripts\python -m server --download
```

## 켜기

| 할 일 | 방법 |
|---|---|
| 서버 켜기 | `run-server.bat` → `http://127.0.0.1:8000`. 같은 PC의 에뮬레이터 앱은 `http://10.0.2.2:8000`으로 붙어요 |
| 화면만 시험 | `run-server-mock.bat` (녹음 내용과 상관없이 정해진 가사·노래를 줘요) |
| 바깥에서 붙게 하기 | `share-server.bat` → 팀 키(`KR_API_KEY`)가 없으면 만들고, Cloudflare 임시 https 주소를 열어요. 주소는 `server-url.txt`에 남고, 창을 닫거나 다시 열면 바뀌어요 |
| 상태·명세 화면 | 주소 뒤에 `/v1/health`, `/docs`(Swagger) |

- 바깥 주소로 오는 요청은 헤더 `X-Api-Key: <팀 키>`가 있어야 해요. 같은 PC에서 바로 온 요청(에뮬레이터)은 키 없이 받아요.
- 앱은 `local.properties`의 `kidsrhythm.serverUrl`, `kidsrhythm.apiKey`로 서버 주소와 팀 키를 정해요.

## 시험

```bat
rem 시험용 목소리 녹음 만들기 (agent\files에 생겨요)
ai\.venv\Scripts\python -m server.tools make-test-audio
rem 켜진 서버에 앱처럼 요청해 보기: 목소리 등록 → 녹음 세션 → 완성 → mp3 받기 → 지우기
ai\.venv\Scripts\python -m server.smoketest
rem 바깥 주소로 시험할 때는 팀 키를 ai-keys.env에서 붙여요
ai\.venv\Scripts\python -m server.smoketest --base https://...
rem 횟수 제한·제목 길이 단위 시험 (모델 없이 돌아요)
ai\.venv\Scripts\python -m unittest discover -s tests
rem 키가 되는지, 바깥에서 붙는지 (키 값은 화면에 안 보여 줘요)
ai\.venv\Scripts\python -m server.tools check-keys
ai\.venv\Scripts\python -m server.tools tunnel-check https://...
```

## 심사용 고정 주소로 켜 두기

심사용 APK에는 서버 주소와 팀 키가 들어가요. 1차 결과(11/23), 2차 PT(12/3~4), 전시(2027.02) 동안 그 주소가 계속 살아 있어야 해요. `share-server.bat`의 임시 주소는 다시 켤 때마다 바뀌어서 APK에 넣을 수 없어요.

### 방법 1. ngrok 무료 고정 주소 (돈 안 들고 제일 빠름)

1. [ngrok](https://ngrok.com)에 가입하고 Windows용 ngrok을 설치해요. 대시보드의 Authtoken으로 한 번 등록해요: `ngrok config add-authtoken <토큰>`
2. 대시보드 Domains에서 계정에 붙은 고정 주소(`….ngrok-free.app`)를 확인해요. ngrok을 다시 켜도 이 주소는 그대로예요.
3. `run-server.bat`으로 서버를 켜고, 다른 창에서 그 주소로 열어요.

```bat
ngrok http --url=https://<고정 주소> 8000
```

4. 바깥에서 붙는지 확인해요. 키 없이 401, 팀 키로 201이 나오면 돼요: `ai\.venv\Scripts\python -m server.tools tunnel-check https://<고정 주소>`
5. 앱 `local.properties`의 `kidsrhythm.serverUrl`을 이 주소로, `kidsrhythm.apiKey`는 `ai-keys.env`의 `KR_API_KEY`로 적어요.

- 무료 한도는 한 달에 보내는 데이터 1GB, 요청 2만 번이에요. 녹음 2초 조각과 1초 폴링 때문에 한 곡에 요청이 100번 안팎이라, 한 달에 200곡쯤이 기준이에요. 넘으면 유료 플랜이나 방법 2로 바꿔요.
- 무료 주소를 브라우저로 열면 ngrok 경고 화면이 먼저 나와요. ngrok 안내로는 앱 같은 프로그램 요청에는 영향이 없어요.

### 방법 2. Cloudflare Named Tunnel (도메인이 있으면, 한도 없음)

Cloudflare에 도메인을 하나 올려 두고, `share-server.bat`이 받아 둔 `tools\cloudflared.exe`로 이름 붙은 터널을 만들어요.

```bat
tools\cloudflared.exe tunnel login
tools\cloudflared.exe tunnel create kidsrhythm
tools\cloudflared.exe tunnel route dns kidsrhythm api.<도메인>
tools\cloudflared.exe tunnel run --url http://127.0.0.1:8000 kidsrhythm
```

### 서버 PC 설정

- 절전을 꺼요: 설정 → 시스템 → 전원 → 화면과 절전을 '안 함'으로.
- 재부팅해도 다시 켜지게, 작업 스케줄러에 `run-server.bat`과 ngrok(또는 cloudflared)을 '로그온할 때' 실행으로 넣어요.
- 심사 기간에는 Windows 업데이트 재시작을 미뤄 둬요.
- 노트북·데스크톱 CPU로 돌리면 한 곡에 1분쯤 걸리고, 녹음 중 '들린 단어'가 늦게 떠요 (받아쓰기 한 번에 30초 안팎).

### 클라우드 (참고)

`deploy/huggingface`의 세 파일과 `server/`, `ai/songmaker/`, `docs/`를 한 폴더에 모으면 Hugging Face Docker Space로 올릴 수 있어요. 11/12에 CPU upgrade(8코어)로 올렸을 때 한 곡에 20초 안팎 걸렸고, 지금 그 Space는 멈춰 뒀어요 (켜 두면 시간당 $0.03).

## 노래 만들기 한도

앱에 넣은 팀 키는 APK에서 꺼낼 수 있어서, 키가 새도 바깥 AI 요금이 끝없이 나가지 않게 서버가 노래 수를 묶어요 (`server/limits.py`). 바깥 주소로 온 요청만 세고, 같은 PC의 에뮬레이터는 세지 않아요. 넘으면 `429 RATE_LIMITED`가 가요.

| 설정 (환경 변수, `ai-keys.env`에 적어도 돼요) | 기본값 | 뜻 |
|---|---|---|
| `KR_SONGS_PER_DAY` | 60 | 하루 전체 곡 수 (한국 시간 자정에 다시 셈, 0이면 제한 없음) |
| `KR_SONGS_PER_HOUR_PER_IP` | 10 | 같은 IP에서 한 시간에 만드는 곡 수 (0이면 제한 없음) |

지금 몇 곡 만들었는지는 `/v1/health`의 `limits`에서 봐요. Anthropic·ElevenLabs 콘솔의 월 사용 한도도 같이 걸어 두면 더 안전해요.

## 키 관리

- 키는 `ai-keys.env`에만 둬요. `.gitignore`에 들어 있어서 저장소에 올라가지 않아요. 채팅이나 문서에도 붙이지 마세요.
- 팀 키(`KR_API_KEY`)는 앱 빌드에 들어가니, 팀원에게만 따로 전달해요.
