@echo off
REM install.bat -- Install Python dependencies for Operator 1 on Windows
REM
REM Usage:
REM   install.bat          Install everything (all 4 stages)
REM   install.bat 1        Stage 1 only: Core libraries (~30s)
REM   install.bat 2        Stage 2 only: ML and statistics (~1-2min)
REM   install.bat 3        Stage 3 only: Deep learning + Bayesian (~5-8min)
REM   install.bat 4        Stage 4 only: Data source wrappers (~1-2min)
REM   install.bat venv     Just create the virtual environment
REM   install.bat check    Run import smoke test only
REM
REM Requires Python 3.12+ installed and on PATH.
REM Download from: https://www.python.org/downloads/

setlocal enabledelayedexpansion

set "PYTHON_MIN_MAJOR=3"
set "PYTHON_MIN_MINOR=12"
set "VENV_DIR=%~dp0venv"
set "REPO_DIR=%~dp0"
set "STAGE=%~1"

if "%STAGE%"=="" set "STAGE=all"

REM ---------- Find Python ----------
call :find_python
if errorlevel 1 exit /b 1

REM ---------- Route to the right action ----------
if "%STAGE%"=="all" goto :install_all
if "%STAGE%"=="1" goto :install_one
if "%STAGE%"=="2" goto :install_two
if "%STAGE%"=="3" goto :install_three
if "%STAGE%"=="4" goto :install_four
if "%STAGE%"=="venv" goto :install_venv
if "%STAGE%"=="check" goto :install_check
if "%STAGE%"=="test" goto :install_check
if "%STAGE%"=="smoke" goto :install_check

echo Usage: install.bat [all^|1^|2^|3^|4^|venv^|check]
echo.
echo   all   -- Install everything (default)
echo   1     -- Stage 1: Core libraries (~30s)
echo   2     -- Stage 2: ML ^& statistics (~1-2min)
echo   3     -- Stage 3: Deep learning + Bayesian (~5-8min)
echo   4     -- Stage 4: Data source wrappers (~1-2min)
echo   venv  -- Just create the virtual environment
echo   check -- Run import smoke test
exit /b 1

REM ========================================================================
REM  Find a suitable Python 3.12+ on the system
REM ========================================================================
:find_python
set "PYTHON_BIN="

REM Try common Python commands
for %%P in (python python3 py) do (
    where %%P >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%V in ('%%P -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                if %%A GEQ %PYTHON_MIN_MAJOR% (
                    if %%B GEQ %PYTHON_MIN_MINOR% (
                        set "PYTHON_BIN=%%P"
                        echo [INFO]  Found Python %%V via %%P
                        goto :find_python_done
                    )
                )
            )
        )
    )
)

REM Try the Python Launcher (py -3.12)
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_BIN=py -3.12"
        echo [INFO]  Found Python 3.12 via py launcher
        goto :find_python_done
    )
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_BIN=py -3.13"
        echo [INFO]  Found Python 3.13 via py launcher
        goto :find_python_done
    )
)

echo [ERROR] Python 3.12 or newer not found on PATH.
echo.
echo Please install Python 3.12+ from:
echo   https://www.python.org/downloads/
echo.
echo IMPORTANT: During installation, check the box that says
echo   "Add Python to PATH"
echo.
echo If Python is already installed but not on PATH, try:
echo   py -3.12 --version
echo or add Python to your PATH manually.
exit /b 1

:find_python_done
exit /b 0

REM ========================================================================
REM  Create / verify virtual environment
REM ========================================================================
:setup_venv
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
    if not errorlevel 1 (
        echo [INFO]  Virtual environment already exists at %VENV_DIR%
        goto :setup_venv_activate
    )
    echo [WARN]  Existing venv is broken. Recreating ...
    rmdir /s /q "%VENV_DIR%"
)

echo [INFO]  Creating virtual environment at %VENV_DIR% ...
%PYTHON_BIN% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo         Make sure python3-venv or the venv module is installed.
    exit /b 1
)

:setup_venv_activate
set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "VPYTHON=%VENV_DIR%\Scripts\python.exe"

echo [INFO]  Upgrading pip ...
"%VPYTHON%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARN]  pip upgrade failed, continuing with existing version
)

for /f "tokens=*" %%V in ('"%VPYTHON%" --version 2^>nul') do echo [INFO]  Python: %%V
for /f "tokens=*" %%V in ('"%PIP%" --version 2^>nul') do echo [INFO]  Pip: %%V
exit /b 0

REM ========================================================================
REM  Stage installers
REM ========================================================================
:stage1
echo [INFO]  Stage 1/4: Core libraries (numpy, pandas, scipy, matplotlib, pytest) ...
"%PIP%" install --timeout 300 -r "%REPO_DIR%requirements\stage1-core.txt"
if errorlevel 1 (
    echo [ERROR] Stage 1 failed. Check the error above and retry: install.bat 1
    exit /b 1
)
echo [INFO]  Stage 1 complete.
exit /b 0

:stage2
echo [INFO]  Stage 2/4: ML ^& statistics (scikit-learn, xgboost, statsmodels, shap) ...
"%PIP%" install --timeout 300 -r "%REPO_DIR%requirements\stage2-ml.txt"
if errorlevel 1 (
    echo [ERROR] Stage 2 failed. Check the error above and retry: install.bat 2
    exit /b 1
)
echo [INFO]  Stage 2 complete.
exit /b 0

:stage3
echo [INFO]  Stage 3/4: Deep learning ^& Bayesian (PyTorch ~2 GB, PyMC) ...
echo [INFO]  This stage downloads large files. If it times out, re-run: install.bat 3
"%PIP%" install --timeout 300 -r "%REPO_DIR%requirements\stage3-deeplearning.txt"
if errorlevel 1 (
    echo [ERROR] Stage 3 failed. This is often due to download timeouts.
    echo         Re-run: install.bat 3
    echo         Or skip this stage -- the pipeline works without deep learning.
    exit /b 1
)
echo [INFO]  Stage 3 complete.
exit /b 0

:stage4
echo [INFO]  Stage 4/4: Data source wrappers (edgartools, yfinance, dart-fss, etc.) ...
"%PIP%" install --timeout 300 -r "%REPO_DIR%requirements\stage4-wrappers.txt"
if errorlevel 1 (
    echo [ERROR] Stage 4 failed. Check the error above and retry: install.bat 4
    exit /b 1
)
echo [INFO]  Stage 4 complete.
exit /b 0

REM ========================================================================
REM  Smoke test
REM ========================================================================
:smoke_test
echo [INFO]  Running import smoke test ...
"%VPYTHON%" -c "import sys; print(f'Python {sys.version}'); exec(\"try:\n import numpy, pandas, scipy, requests, yaml, matplotlib\n print('[OK] Stage 1: Core libraries')\nexcept ImportError as e:\n print(f'[SKIP] Stage 1: {e}')\ntry:\n import sklearn, statsmodels, xgboost, ruptures, hmmlearn, arch, shap, mapie, dtaidistance\n print('[OK] Stage 2: ML and statistics')\nexcept ImportError as e:\n print(f'[SKIP] Stage 2: {e}')\ntry:\n import torch, arviz, pymc\n print('[OK] Stage 3: Deep learning and Bayesian')\nexcept ImportError as e:\n print(f'[SKIP] Stage 3: {e}')\ntry:\n import edgar, yfinance, wbgapi, fredapi\n print('[OK] Stage 4: Data source wrappers')\nexcept ImportError as e:\n print(f'[SKIP] Stage 4: {e}')\nprint()\nprint('Import test complete. Any [SKIP] stages can be installed with: install.bat <stage_number>')\")"
echo [INFO]  Smoke test complete.
exit /b 0

REM ========================================================================
REM  Main flows
REM ========================================================================
:install_all
call :setup_venv
if errorlevel 1 exit /b 1
call :stage1
if errorlevel 1 exit /b 1
call :stage2
if errorlevel 1 exit /b 1
call :stage3
if errorlevel 1 exit /b 1
call :stage4
if errorlevel 1 exit /b 1
call :smoke_test
echo.
echo [INFO]  All done. Activate the virtual environment with:
echo [INFO]      venv\Scripts\activate
echo [INFO]  Then run: python run.py
goto :eof

:install_one
call :setup_venv
if errorlevel 1 exit /b 1
call :stage1
goto :eof

:install_two
call :setup_venv
if errorlevel 1 exit /b 1
call :stage2
goto :eof

:install_three
call :setup_venv
if errorlevel 1 exit /b 1
call :stage3
goto :eof

:install_four
call :setup_venv
if errorlevel 1 exit /b 1
call :stage4
goto :eof

:install_venv
call :setup_venv
if errorlevel 1 exit /b 1
echo [INFO]  Virtual environment ready. Activate with: venv\Scripts\activate
goto :eof

:install_check
call :setup_venv
if errorlevel 1 exit /b 1
call :smoke_test
goto :eof
