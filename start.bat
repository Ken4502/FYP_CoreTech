@echo off
setlocal enabledelayedexpansion
title FYP CoreTech AI Suite

echo =======================================================================
echo          FYP CoreTech: One-Click Plug ^& Play AI Launcher
echo =======================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Check and Pull Git LFS Model Weights
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] Checking model weights with Git LFS...
    git lfs pull
    echo.
)

:: 3. Setup Virtual Environment
if not exist ".venv" (
    echo [*] First-time setup detected: Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    call .venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Some dependencies had installation warnings. Continuing...
    )
    echo.
) else (
    call .venv\Scripts\activate.bat
)

:: 4. Launch Services
echo [*] Launching FloraNet-250 (Port 8000) and PhytoCare Ultra (Port 8001)...
python launch.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Server session ended.
    pause
)

