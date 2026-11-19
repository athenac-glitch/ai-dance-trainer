@echo off
chcp 65001 >nul
title AI K-Pop Dance Trainer Launcher (BA5077 Edge AIoT)
cd /d "%~dp0"

echo =================================================================
echo   AI K-Pop Dance Trainer (BA5077 Edge AIoT)
echo   Student: Yu-Ting Chang (113401532)
echo =================================================================
echo.
echo Starting GUI Launcher (SOURCE\launcher.py)...
echo.

python SOURCE\launcher.py
if errorlevel 1 (
    echo Retrying with direct local Python path...
    "%LOCALAPPDATA%\Python\pythoncore-3.15-64\python.exe" SOURCE\launcher.py
)

if errorlevel 1 (
    echo.
    echo [Error] Failed to start Python. Please check your Python installation.
    pause
)
