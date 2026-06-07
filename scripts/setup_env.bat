@echo off
REM Create a Python virtual environment named 'psformer-env' and install all
REM dependencies needed to run the PSformer reproduction (training + inference).
REM
REM Usage (from Command Prompt, double-click also works):
REM     scripts\setup_env.bat
REM
REM Then activate it (see the message printed at the end):
REM     psformer-env\Scripts\activate.bat
REM
REM Requires Python 3.10+ (the code uses `X | None` type-hint syntax).

setlocal enableextensions

set "ENV_NAME=psformer-env"

REM Move to the repo root (parent of this script's directory) so the env is
REM created at the top level regardless of where the script is called from.
pushd "%~dp0.."

REM Pick a Python interpreter (override with: set PYTHON=C:\path\to\python.exe).
if "%PYTHON%"=="" set "PYTHON=python"
where %PYTHON% >nul 2>&1
if errorlevel 1 (
    echo ERROR: '%PYTHON%' not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/ or set PYTHON to its path.
    popd & exit /b 1
)

REM Verify Python ^>= 3.10.
%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10+ is required.
    %PYTHON% --version
    echo Install a newer Python or set PYTHON to a 3.10+ interpreter and re-run.
    popd & exit /b 1
)
for /f "delims=" %%V in ('%PYTHON% --version') do echo Using %%V

REM Create the venv (skip if it already exists).
if exist "%ENV_NAME%\Scripts\python.exe" (
    echo ^>^> '%ENV_NAME%' already exists -- reusing it.
) else (
    echo ^>^> Creating virtual environment '%ENV_NAME%'...
    %PYTHON% -m venv "%ENV_NAME%"
    if errorlevel 1 ( echo ERROR: failed to create venv. & popd & exit /b 1 )
)

REM Install dependencies into the env without needing to 'activate' first.
echo ^>^> Upgrading pip and installing dependencies...
"%ENV_NAME%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 ( echo ERROR: pip upgrade failed. & popd & exit /b 1 )
"%ENV_NAME%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: dependency install failed. & popd & exit /b 1 )

echo.
echo ========================================================================
echo Done. Environment '%ENV_NAME%' is ready.
echo.
echo Activate it with:
echo     %ENV_NAME%\Scripts\activate.bat
echo.
echo Then verify the install with the parameter-count check:
echo     python -m experiments.test_param_count
echo.
echo Deactivate later with:  deactivate
echo ========================================================================

popd
endlocal
