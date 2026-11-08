@echo off
setlocal enabledelayedexpansion
title UrbanGreen AI Core Services (FloraNet-261 & PhytoCare Ultra)

echo =======================================================================
echo     UrbanGreen CoreTech: One-Click Local AI Inference Launcher
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
    if not exist ".venv" (
        echo [*] Setting up local virtual environment in .venv...
        python -m venv .venv
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to create virtual environment!
            pause
            exit /b 1
        )
        call .venv\Scripts\activate.bat
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    )
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
)

REM 3. Check and Pull Git LFS Model Weights
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Verifying model weights with Git LFS...
    git lfs pull
    echo.
)

REM 4. Launch Services
echo [*] Launching FloraNet-261 on Port 8000 and PhytoCare Ultra on Port 8001...
echo [*] Python executable: !PYTHON_EXE!
echo.
"!PYTHON_EXE!" launch.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Server session ended with code %errorlevel%.
    pause
)
