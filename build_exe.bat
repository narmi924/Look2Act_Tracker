@echo off
setlocal

cd /d "%~dp0"

set CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe
set ENV_NAME=gaze-env
set SPEC_FILE=Look2Act.spec
set OUTPUT_EXE=dist\Look2Act\Look2Act.exe

if not exist "%CONDA_EXE%" (
    echo Conda not found: %CONDA_EXE%
    exit /b 1
)

if not exist "%SPEC_FILE%" (
    echo Spec file not found: %SPEC_FILE%
    exit /b 1
)

echo Project root: %CD%
echo Input: %SPEC_FILE%, main.py, configs\, checkpoints\, Look2Act.ico
echo Output: %OUTPUT_EXE%
echo.
echo [1/2] Building Look2Act folder EXE...
"%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python -m PyInstaller --clean --noconfirm "%SPEC_FILE%"
if errorlevel 1 exit /b 1

if not exist "%OUTPUT_EXE%" (
    echo Build finished but output EXE was not found: %OUTPUT_EXE%
    exit /b 1
)

echo [2/2] Build complete: %OUTPUT_EXE%
exit /b 0
