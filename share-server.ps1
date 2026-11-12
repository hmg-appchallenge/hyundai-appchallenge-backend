# share-server.ps1 - 이 PC의 노래 생성 서버를 임시 https 주소로 열어서 팀원이 붙을 수 있게 한다.
# Cloudflare 임시 터널(cloudflared, 계정 필요 없음)을 쓴다. 이 창을 닫으면 주소도 없어지고, 다시 열면 주소가 바뀐다.
# 바깥에서 오는 요청은 팀 키(KR_API_KEY)가 있어야 받는다. 같은 PC의 에뮬레이터는 지금처럼 키 없이 된다.
param([int]$Port = 8000)
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$root = $PSScriptRoot
$keysFile = Join-Path $root 'ai-keys.env'
$toolsDir = Join-Path $root 'tools'
$exe = Join-Path $toolsDir 'cloudflared.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir, $toolsDir | Out-Null
$Host.UI.RawUI.WindowTitle = '서버 임시 공개 주소 (이 창을 닫으면 주소가 없어져요)'

function Stop-Here([string]$msg) {
    Write-Host ''
    Write-Host "[멈춤] $msg" -ForegroundColor Yellow
    pause
    exit 1
}

function Get-Health([string]$base, [int]$timeout = 5, [int]$tries = 3) {
    for ($i = 0; $i -lt $tries; $i++) {
        try { return Invoke-RestMethod -Uri "$base/v1/health" -TimeoutSec $timeout -UseBasicParsing } catch { Start-Sleep -Milliseconds 700 }
    }
    return $null
}

# ---------- 1. 팀 키 ----------
$key = $null
if (Test-Path $keysFile) {
    $line = Get-Content $keysFile -Encoding UTF8 | Where-Object { $_ -match '^\s*KR_API_KEY\s*=\s*\S' } | Select-Object -First 1
    if ($line) { $key = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'") }
}
if (-not $key) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = 'kr-' + (([Convert]::ToBase64String($bytes) -replace '[+/=]', '').Substring(0, 24))
    $text = "`r`n# 팀원이 임시 공개 주소로 붙을 때 쓰는 키 (share-server.bat이 만듦)`r`nKR_API_KEY=$key`r`n"
    [System.IO.File]::AppendAllText($keysFile, $text, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host '팀 키를 새로 만들어서 ai-keys.env에 적었어요.' -ForegroundColor Green
}

# ---------- 2. 서버가 팀 키를 쓰는 상태로 켜져 있는지 ----------
# 키 없이 켜진 서버를 바깥에 열면 누구나 노래를 만들 수 있으니, 키를 쓰는 서버가 뜰 때까지 기다린다.
$local = "http://127.0.0.1:$Port"
$h = Get-Health $local
$said = $false
while (-not ($h -and $h.auth -eq $true)) {
    if (-not $said) {
        Write-Host ''
        if (-not $h) {
            Write-Host '노래 생성 서버가 아직 안 켜져 있어요. run-server.bat을 켜 주세요.' -ForegroundColor Yellow
        } else {
            Write-Host '서버가 팀 키 없이 켜져 있어요. 서버 창을 닫고 run-server.bat을 다시 켜 주세요 (키를 새로 읽어야 해요).' -ForegroundColor Yellow
        }
        Write-Host '이 창은 그대로 두세요. 서버가 준비되면 이어서 진행해요...'
        $said = $true
    }
    Start-Sleep -Seconds 3
    $h = Get-Health $local 3
}
Write-Host "서버 확인: 팀 키 사용 중, 모델 준비 $(if ($h.ready) { '끝' } else { '중 (1~2분)' })" -ForegroundColor Green

# ---------- 3. cloudflared 준비 ----------
if (-not (Test-Path $exe)) {
    Write-Host ''
    Write-Host 'Cloudflare의 임시 주소 도구(cloudflared.exe, 약 60MB)가 필요해요.'
    Write-Host '받는 곳: https://github.com/cloudflare/cloudflared/releases (Cloudflare 공식 배포)'
    $a = Read-Host '받을까요? (y/n)'
    if ($a -notmatch '^\s*(y|yes)\s*$') { Stop-Here '받지 않아서 멈췄어요.' }
    $url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
    $part = "$exe.part"
    & curl.exe -L --fail --retry 3 --retry-delay 2 -o $part $url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $part) -or (Get-Item $part).Length -lt 10MB) {
        Stop-Here '받지 못했어요. 인터넷 연결을 확인하고 다시 실행해 주세요.'
    }
    Move-Item -Force $part $exe
    $sig = Get-AuthenticodeSignature $exe
    $signer = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { '' }
    if ($sig.Status -ne 'Valid' -or $signer -notmatch 'Cloudflare') {
        Write-Host "서명 확인: $($sig.Status) $signer" -ForegroundColor Yellow
        $a = Read-Host 'Cloudflare 서명을 확인하지 못했어요. 그래도 쓸까요? (y/n)'
        if ($a -notmatch '^\s*(y|yes)\s*$') {
            Move-Item -Force $exe "$exe.unverified"
            Stop-Here '쓰지 않았어요. tools\cloudflared.exe.unverified 로 옮겨 뒀어요.'
        }
    } else {
        Write-Host '서명 확인: Cloudflare 공식 파일이에요.' -ForegroundColor Green
    }
}

# ---------- 4. 임시 주소 열기 ----------
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$tlog = Join-Path $logDir "tunnel-$stamp.log"
Write-Host ''
Write-Host '임시 주소를 여는 중...'
$p = Start-Process -FilePath $exe -ArgumentList @('tunnel', '--no-autoupdate', '--url', $local) `
    -RedirectStandardError $tlog -RedirectStandardOutput "$tlog.out" -NoNewWindow -PassThru
$public = $null
for ($i = 0; $i -lt 60 -and -not $public; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) { break }
    if (Test-Path $tlog) {
        $m = Select-String -Path $tlog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches |
            ForEach-Object { $_.Matches } | ForEach-Object { $_.Value } | Where-Object { $_ -notmatch '//api\.' } |
            Select-Object -First 1
        if ($m) { $public = $m }
    }
}
if (-not $public) {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if (Test-Path $tlog) { Get-Content $tlog -Tail 15 | ForEach-Object { Write-Host "  $_" } }
    Stop-Here "임시 주소를 열지 못했어요. 기록: $tlog"
}

Set-Content -Path (Join-Path $root 'server-url.txt') -Value $public -Encoding ASCII
$msg = "kidsrhythm.serverUrl=$public`r`nkidsrhythm.apiKey=$key"
try { Set-Clipboard -Value $msg } catch {}

Write-Host ''
Write-Host '========================================================' -ForegroundColor Cyan
Write-Host " 임시 주소: $public" -ForegroundColor Cyan
Write-Host '========================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host '팀원은 프론트 local.properties에 아래 두 줄을 넣고 앱을 다시 빌드하면 돼요 (클립보드에 복사해 뒀어요):'
Write-Host ''
Write-Host $msg
Write-Host ''
Write-Host '- 이 창과 서버 창(run-server.bat)을 닫으면 팀원도 못 붙어요. PC가 잠들지 않게 해 주세요.'
Write-Host '- 이 창을 다시 열면 주소가 바뀌어요. 키는 그대로예요.'
Write-Host '- 이 PC의 에뮬레이터는 지금처럼 http://10.0.2.2:8000 으로 키 없이 돼요.'
Write-Host ''

# 바깥에서 실제로 붙는지 확인 (주소가 퍼지는 데 몇 초 걸린다). 45초까지만 본다
Write-Host '바깥에서 접속되는지 확인하는 중...'
$ok = $null
$deadline = (Get-Date).AddSeconds(45)
while (-not $ok -and (Get-Date) -lt $deadline -and -not $p.HasExited) {
    Start-Sleep -Seconds 3
    $ok = Get-Health $public 6 1
}
if ($ok) { Write-Host '바깥에서 접속 확인: 됨' -ForegroundColor Green }
else { Write-Host '바깥에서 접속 확인: 아직 안 됨. 1~2분 뒤 브라우저로 주소/v1/health 를 열어 보세요.' -ForegroundColor Yellow }
Write-Host ''
Write-Host '열어 두는 중... (닫으려면 이 창을 닫으세요)'
try {
    if (-not $p.HasExited) { $p.WaitForExit() }
    Write-Host "임시 주소 도구가 꺼졌어요. 기록: $tlog" -ForegroundColor Yellow
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}
pause
