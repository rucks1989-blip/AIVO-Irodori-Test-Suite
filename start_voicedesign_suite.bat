@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "API_URL=http://127.0.0.1:8020/health"
set "UI_URL=http://127.0.0.1:8520"
set "API_RUNNER=%~dp0run_voicedesign_api.bat"
set "UI_RUNNER=%~dp0run_voicedesign_test_ui.bat"
set "MAX_WAIT_SECONDS=120"

call :check_url "%API_URL%"
if "!URL_OK!"=="1" (
  echo [API ready]
) else (
  echo [API starting]
  start "" /min cmd /c call "%API_RUNNER%"
  call :wait_for_url "%API_URL%" "API"
  if "!URL_OK!" NEQ "1" exit /b 1
)

call :check_url "%UI_URL%"
if "!URL_OK!"=="1" (
  echo [UI ready]
) else (
  echo [UI starting]
  start "" /min cmd /c call "%UI_RUNNER%"
  call :wait_for_url "%UI_URL%" "UI"
  if "!URL_OK!" NEQ "1" exit /b 1
)

echo [Opening browser]
start "" "%UI_URL%"
exit /b 0

:wait_for_url
set "WAIT_URL=%~1"
set "WAIT_LABEL=%~2"
set "URL_OK=0"
for /L %%I in (1,1,%MAX_WAIT_SECONDS%) do (
  call :check_url "%WAIT_URL%"
  if "!URL_OK!"=="1" (
    echo [!WAIT_LABEL! ready]
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
echo [!WAIT_LABEL! failed]
exit /b 1

:check_url
set "URL_OK=0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ProgressPreference='SilentlyContinue';" ^
  "try {" ^
  "  $r = Invoke-WebRequest -Uri '%~1' -UseBasicParsing -TimeoutSec 5;" ^
  "  if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 }" ^
  "} catch { exit 1 }" >nul 2>nul
if not errorlevel 1 set "URL_OK=1"
exit /b 0
