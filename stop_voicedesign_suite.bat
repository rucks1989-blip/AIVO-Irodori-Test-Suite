@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

call :stop_port 8020 API
call :stop_port 8520 UI
exit /b 0

:stop_port
set "TARGET_PORT=%~1"
set "TARGET_LABEL=%~2"
set "FOUND_ANY=0"
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":%TARGET_PORT% .*LISTENING"') do (
  if not "%%P"=="0" (
    set "FOUND_ANY=1"
    echo [Stopping %TARGET_LABEL%] port=%TARGET_PORT% pid=%%P
    taskkill /PID %%P /T /F >nul 2>nul
  )
)
if "!FOUND_ANY!"=="0" (
  echo [No process] port=%TARGET_PORT%
)
exit /b 0
