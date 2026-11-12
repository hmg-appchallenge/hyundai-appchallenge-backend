# setup-server.ps1 - 노래 생성 서버 환경을 만든다: (없으면) setup-ai.ps1 → 서버 패키지 → 서버 모델
param([switch]$NoPause)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$root = $PSScriptRoot
$py = Join-Path $root 'ai\.venv\Scripts\python.exe'

function Fail([string]$msg) {
    Write-Host ''
    Write-Host "[멈춤] $msg" -ForegroundColor Yellow
    if (-not $NoPause) { pause }
    exit 1
}

if (-not (Test-Path $py)) {
    Write-Host '[1/3] AI 환경(ai\.venv)이 없어서 setup-ai.ps1부터 실행해요' -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'setup-ai.ps1')
    if (-not (Test-Path $py)) { Fail 'AI 환경을 만들지 못했어요. ai\logs 폴더의 setup-ai 기록을 확인해 주세요.' }
} else {
    Write-Host '[1/3] AI 환경(ai\.venv)이 이미 있어요' -ForegroundColor Cyan
}

Write-Host '[2/3] 서버 패키지 설치' -ForegroundColor Cyan
& $py -m pip install --disable-pip-version-check -r (Join-Path $root 'server\requirements.txt') 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail '서버 패키지 설치가 실패했어요.' }

Write-Host '[3/3] 서버 모델 받기 (화자 구분, 말소리 구간, 음성 인식, TTS)' -ForegroundColor Cyan
Push-Location $root
& $py -m server --download 2>&1 | Out-Host
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail '모델을 받지 못했어요. 인터넷 연결을 확인하고 다시 실행해 주세요.' }

Write-Host ''
Write-Host '서버 준비 완료! ai-keys.env에 키를 채운 뒤 run-server.bat으로 켜요.' -ForegroundColor Green
if (-not $NoPause) { pause }
