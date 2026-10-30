@echo off
REM ==========================================================================
REM  setup.bat  --  run ONCE. Creates python\.venv and installs everything.
REM     1. makes the virtual environment  .venv
REM     2. installs PyTorch WITH GPU support (CUDA 12.6 build, ~2.5 GB download)
REM     3. installs the rest from requirements.txt
REM     4. prints whether PyTorch can see your GPU
REM ==========================================================================
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo ===== creating .venv =====
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python was not found. Install Python 3.10-3.13 from python.org
        echo and tick "Add python.exe to PATH" during the install. Then run this again.
        pause
        exit /b 1
    )
)
set PY=.venv\Scripts\python.exe

echo ===== updating pip =====
%PY% -m pip install --upgrade pip

echo ===== installing PyTorch with CUDA (GPU) support =====
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo PyTorch install failed - see GUIDE_1, Problems.
    pause
    exit /b 1
)

echo ===== installing the other packages =====
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed - see GUIDE_1, Problems.
    pause
    exit /b 1
)

echo.
echo ===== checking the GPU =====
%PY% -c "import torch; ok=torch.cuda.is_available(); print('PyTorch', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) + ', %%.1f GB' %% (torch.cuda.get_device_properties(0).total_memory/1e9) if ok else 'NOT FOUND - training would run on the CPU (very slow). See GUIDE_1, Problems.')"
echo.
echo Done. Next:  smoke_test.bat
pause
