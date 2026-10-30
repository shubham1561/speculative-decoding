@echo off
REM  smoke_test.bat  --  1-minute check that everything works (tiny models, no download)
REM  and then the unit tests. Your real data, checkpoints and results are not touched.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=python

echo ===== 1/2 SMOKE TEST =====
%PY% smoke_test.py
if errorlevel 1 (
    echo The smoke test failed - read the [FAIL] lines above.
    pause
    exit /b 1
)
echo.
echo ===== 2/2 UNIT TESTS =====
%PY% -m pytest
pause
