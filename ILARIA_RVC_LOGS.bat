@echo off
title ILARIA_RVC
cd /d "%~dp0"
if exist ilaria_launcher.py (
    runtime\python.exe ilaria_launcher.py --pycmd runtime\python.exe --port 6969
) else (
    runtime\python.exe ilaria\ilaria_launcher.py --pycmd runtime\python.exe --port 6969
)
if errorlevel 1 pause
