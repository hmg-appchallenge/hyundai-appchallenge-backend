# run-server.ps1 - 노래 생성 서버를 켠다. 에뮬레이터 안의 앱은 http://10.0.2.2:8000 으로 접속한다.
# real: AI 모델로 진짜 노래를 만든다 / mock: 모델 없이 정해진 응답을 준다 (앱 연동 테스트용)
param(
    [ValidateSet('real', 'mock')]
    [string]$Mode = 'real',
    [int]$Port = 8000,
    [switch]$NoPause
)
# 'Continue'인 이유: 파이썬 경고가 stderr로 나와도 창이 멈추지 않게 한다.
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$py = Join-Path $root 'ai\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host '[멈춤] AI 환경이 없어. setup-ai.bat 을 먼저 실행해 줘.' -ForegroundColor Yellow
    if (-not $NoPause) { pause }
    exit 1
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("server-{0}-{1}.log" -f $Mode, (Get-Date -Format 'yyyyMMdd-HHmmss'))
$Host.UI.RawUI.WindowTitle = "노래 생성 서버 ($Mode) - 포트 $Port"
Write-Host "노래 생성 서버($Mode)를 켜는 중이야. 이 창을 닫으면 서버도 꺼져." -ForegroundColor Green
Write-Host "로그: $log"
Push-Location $root
try { & $py -m server --mode $Mode --port $Port --log $log } finally { Pop-Location }
Write-Host '서버가 꺼졌어.'
if (-not $NoPause) { pause }
