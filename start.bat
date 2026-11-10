@echo off
setlocal enabledelayedexpansion
title UrbanGreen AI Core Services (FloraNet-261 & PhytoCare Ultra)

echo =======================================================================
echo          UrbanGreen CoreTech: Local & Online AI Launcher
echo =======================================================================
echo.

REM 1. Set working directory to CoreTech
cd /d "%~dp0"
if exist "CoreTech\launch.py" (
    cd /d "%~dp0CoreTech"
)

REM 2. Detect Python / Virtual Environment
set "PYTHON_EXE="

if exist "Plant_Identification\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\Plant_Identification\.venv\Scripts\python.exe"
    echo [*] Detected workstation environment: Plant_Identification\.venv
) else if exist "..\CoreTech\Plant_Identification\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\..\CoreTech\Plant_Identification\.venv\Scripts\python.exe"
    echo [*] Detected workstation environment: CoreTech\Plant_Identification\.venv
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    echo [*] Detected virtual environment: .venv
)

if "!PYTHON_EXE!"=="" (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python was not found on your system PATH!
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

REM 3. Check and Pull Git LFS Model Weights
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Verifying model weights with Git LFS...
    git lfs pull
    echo.
)

REM 4. Mode Selection
set "EXTRA_FLAGS="
if "%1"=="" (
    echo Select AI Hosting Mode:
    echo   [1] Online Cloudflare Tunnel (Free Public HTTPS - Accessible Anywhere on Phone)
    echo   [2] Local Mode Only (Localhost ^& Home Wi-Fi LAN)
    echo.
    set /p "CHOICE=Enter choice [1 or 2, default is 1]: "
    if "!CHOICE!"=="2" (
        set "EXTRA_FLAGS="
        echo [*] Selected Local Mode.
    ) else (
        set "EXTRA_FLAGS=--tunnel"
        echo [*] Selected Online Cloudflare Tunnel Mode.
    )
    echo.
) else (
    set "EXTRA_FLAGS=%*"
)

REM 5. Launch Services
echo [*] Launching FloraNet-261 on Port 8000 and PhytoCare Ultra on Port 8001...
echo [*] Python executable: !PYTHON_EXE!
echo.
"!PYTHON_EXE!" launch.py !EXTRA_FLAGS!

if %errorlevel% neq 0 (
    echo.
    echo [!] Server session ended with code %errorlevel%.
    pause
)
