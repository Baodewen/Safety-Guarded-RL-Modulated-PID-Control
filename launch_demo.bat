@echo off
setlocal

set "ROOT=%~dp0"
set "HOST=127.0.0.1"
set "PORT=8000"
set "URL=http://%HOST%:%PORT%/"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install Python 3.10+ and make sure "python" works in a new terminal.
  exit /b 1
)

python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is installed but failed to start.
  exit /b 1
)

python -c "import socket; s=socket.socket(); s.bind(('%HOST%', %PORT%)); s.close()" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Port %PORT% is already in use.
  echo Change the port in launch_demo.bat or stop the conflicting process.
  exit /b 1
)

echo Starting demo server on %URL%
start "Safety-Guarded RL-PID Demo Server" /D "%ROOT%" python demo_server.py --host %HOST% --port %PORT%

powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(10); $ok=$false; while((Get-Date) -lt $deadline){ try { Invoke-WebRequest -UseBasicParsing '%URL%' | Out-Null; $ok=$true; break } catch { Start-Sleep -Milliseconds 400 } }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo [ERROR] Demo server did not become ready within 10 seconds.
  exit /b 1
)

start "" "%URL%"
echo Demo opened in your default browser.
exit /b 0
