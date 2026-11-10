@echo off
setlocal enabledelayedexpansion
title UrbanGreen AI Online Server (Cloudflare Public HTTPS Tunnel)

echo =======================================================================
echo     UrbanGreen CoreTech: One-Click Online AI Cloudflare Tunnel
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

REM 4. Launch Services with Cloudflare Tunnel
echo [*] Launching FloraNet-261 + PhytoCare Ultra + Cloudflare Tunnel...
echo [*] Python executable: !PYTHON_EXE!
echo.
"!PYTHON_EXE!" launch.py --tunnel %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Server session ended with code %errorlevel%.
    pause
)
