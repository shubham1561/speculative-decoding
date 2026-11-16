@echo off
REM ==========================================================================
REM  run_gpt2.bat  --  double-click to run ONLY the GPT-2 part (steps 6-7):
REM     6  distil distilgpt2 towards gpt2-large   (downloads ~3.6 GB once)
REM     7  benchmark the gpt2 pair on story and general prompts
REM  Your own models are not touched. Takes about 1.5-2 hours.
REM  Log: logs\overnight_<date>_<time>.log     Table: results\RESULTS.md
REM ==========================================================================
cd /d "%~dp0"

echo Checking the internet connection to huggingface.co ...
curl -s -o NUL --max-time 15 https://huggingface.co
if errorlevel 1 (
    echo.
    echo Cannot reach huggingface.co - check the internet, then double-click this again.
    pause
    exit /b 1
)
echo OK.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_overnight.ps1" -OnlyGpt2
echo.
pause
