# make-song.ps1 - 녹음 파일로 노래 샘플을 만든다.
# 사용법: 녹음 파일(최대 3개)을 make-song.bat 위에 끌어다 놓거나, ai\input 폴더에 넣고 더블클릭.
# 녹음이 하나도 없으면 시험용 가짜 목소리로 만든다.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Files)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$ai = Join-Path $PSScriptRoot 'ai'
$vpy = Join-Path $ai '.venv\Scripts\python.exe'
if (-not (Test-Path $vpy)) {
    Write-Host '[멈춤] AI 환경이 아직 없어. setup-ai.bat 을 먼저 실행해 줘.' -ForegroundColor Yellow
    exit 1
}
$audioExt = @('.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.3gp', '.amr', '.wma', '.mp4', '.webm', '.caf')
$inputs = @($Files | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
if ($inputs.Count -eq 0) {
    $inputs = @(Get-ChildItem -LiteralPath (Join-Path $ai 'input') -File -ErrorAction SilentlyContinue |
        Where-Object { $audioExt -contains $_.Extension.ToLower() } | Sort-Object Name | ForEach-Object { $_.FullName })
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Push-Location $ai
$env:PYTHONPATH = $ai
if ($inputs.Count -eq 0) {
    $out = Join-Path $ai "output\$stamp-selftest"
    Write-Host '녹음 파일이 없어서 시험용 가짜 목소리로 만들게.' -ForegroundColor Cyan
    Write-Host '먼저 TTS 목소리 10개 미리 듣기 파일을 만들고, 이어서 노래 6곡을 만들어.'
    & $vpy -m songmaker --voices --out $out
    & $vpy -m songmaker --selftest --out $out
} else {
    if ($inputs.Count -gt 3) {
        Write-Host '녹음은 3개까지만 써. 앞의 3개만 쓸게.' -ForegroundColor Yellow
        $inputs = $inputs[0..2]
    }
    $out = Join-Path $ai "output\$stamp"
    Write-Host ("녹음 {0}개로 만들게: {1}" -f $inputs.Count, (($inputs | ForEach-Object { Split-Path $_ -Leaf }) -join ', ')) -ForegroundColor Cyan
    & $vpy -m songmaker --out $out @inputs
}
$code = $LASTEXITCODE
Pop-Location
Write-Host ''
if ($code -eq 0 -and (Test-Path $out)) {
    Write-Host '완성! 결과 폴더를 열게. mp3를 들어 보고 summary.txt에서 가사를 확인해 줘.' -ForegroundColor Green
    Start-Process explorer.exe $out
} else {
    Write-Host '[멈춤] 중간에 멈췄어. ai\logs 폴더의 기록을 확인해 줘.' -ForegroundColor Yellow
}
