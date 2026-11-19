@echo off
setlocal
if not exist "%~dp0app\runtime\python.exe" (
    echo Python runtime missing. Extract the full Ilaria RVC Windows ZIP.
    pause
    exit /b 1
)
call "%~dp0app\ILARIA_RVC_LOGS.bat"
exit /b %errorlevel%
