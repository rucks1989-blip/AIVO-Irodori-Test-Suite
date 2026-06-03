@echo off
setlocal
set PYTHONUTF8=1
set "PATH=%~dp0vendor\Irodori-TTS\ffmpeg\bin;%PATH%"
call "%~dp0.venv\Scripts\activate.bat"
python -m uvicorn irodori_voicedesign_api:app --host 127.0.0.1 --port 8020
