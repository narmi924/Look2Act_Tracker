@echo off
setlocal

set CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe
set ENV_NAME=gaze-env

if not exist "%CONDA_EXE%" (
    echo Conda not found: %CONDA_EXE%
    exit /b 1
)

echo [1/2] Building Look2Act folder EXE...
"%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python -m PyInstaller --clean --noconfirm Look2Act.spec
if errorlevel 1 exit /b 1

echo [2/2] Build complete: dist\Look2Act\Look2Act.exe
exit /b 0
