@echo off
setlocal
call "%~dp0.venv\Scripts\activate.bat"
python -m streamlit run "%~dp0irodori_voicedesign_test_ui.py" --server.headless true --server.port 8520
