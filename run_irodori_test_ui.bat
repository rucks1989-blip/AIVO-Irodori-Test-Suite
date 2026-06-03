@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Python virtual environment not found at %~dp0.venv\Scripts\python.exe
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0irodori_test_ui.py" --server.headless true
