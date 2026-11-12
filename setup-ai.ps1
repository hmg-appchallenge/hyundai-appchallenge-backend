# setup-ai.ps1 - AI 노래 시제품 환경을 만든다: 파이썬 가상환경 + 패키지 + 모델(약 2.5GB)
param([string]$SttModel = 'large-v3-turbo')
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$ai = Join-Path $PSScriptRoot 'ai'
$venv = Join-Path $ai '.venv'
$vpy = Join-Path $venv 'Scripts\python.exe'
$logDir = Join-Path $ai 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Transcript -Path (Join-Path $logDir "setup-ai-$stamp.log") | Out-Null

function Fail([string]$msg) {
    Write-Host ''
    Write-Host "[멈춤] $msg" -ForegroundColor Yellow
    Write-Host 'ai\logs 폴더의 setup-ai 기록을 확인해 줘.'
    Stop-Transcript | Out-Null
    exit 1
}

function Find-Python {
    foreach ($v in '312', '311') {
        $p = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"
        if (Test-Path $p) { return $p }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in '3.12', '3.11', '3.13') {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe)) { return $exe }
        }
    }
    $p = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    if (Test-Path $p) { return $p }
    return $null
}

Write-Host '[1/4] 공간 확인'
$free = [math]::Round((Get-PSDrive -Name C).Free / 1GB, 1)
Write-Host "  C 드라이브 여유: $free GB (필요: 약 4GB)"
if ($free -lt 4) { Fail '공간이 모자라. 4GB 이상 비우고 다시 실행해 줘.' }

Write-Host '[2/4] 파이썬 찾기'
$python = Find-Python
if (-not $python) { Fail 'Python 3.12를 못 찾았어.' }
Write-Host "  사용할 파이썬: $python"
& $python --version

if (-not (Test-Path $vpy)) {
    Write-Host '  가상환경 만드는 중 (ai\.venv)'
    & $python -m venv $venv 2>&1 | Out-Host
    if (-not (Test-Path $vpy)) { Fail '가상환경을 만들지 못했어.' }
}

Write-Host '[3/4] 패키지 설치 (처음엔 몇 분 걸려)'
& $vpy -m pip install --disable-pip-version-check --upgrade pip 2>&1 | Out-Host
& $vpy -m pip install --disable-pip-version-check -r (Join-Path $ai 'requirements.txt') 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail '패키지 설치가 실패했어.' }

Write-Host '[4/4] 모델 내려받기 (음성 인식 약 1.6GB + TTS)'
Push-Location $ai
$env:PYTHONPATH = $ai
& $vpy -m songmaker --download --stt-model $SttModel 2>&1 | Out-Host
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { Fail '모델을 받지 못했어. 인터넷 연결을 확인하고 다시 실행해 줘.' }

Write-Host ''
Write-Host 'AI 환경 준비 완료!' -ForegroundColor Green
Write-Host '다음: make-song.bat 을 더블클릭해. 녹음이 아직 없으면 시험용 목소리로 먼저 만들어 줘.'
Stop-Transcript | Out-Null
