@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make-song.ps1" %*
pause
