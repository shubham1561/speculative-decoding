@echo off
REM ==========================================================================
REM  START_DEMO.bat  --  double-click to open the speculative decoding demo.
REM  Starts a small local server (only on this computer), loads the models
REM  and opens the demo page in your browser: http://127.0.0.1:8000
REM  Keep this window open while you use the demo. Close it to stop.
REM ==========================================================================
cd /d "%~dp0python"
if not exist .venv\Scripts\python.exe (
    echo The Python environment is missing. Run python\setup.bat first ^(GUIDE_1^).
    pause
    exit /b 1
)
title Speculative decoding demo - close this window to stop
.venv\Scripts\python.exe demo_server.py %*
echo.
echo The demo server has stopped.
pause
