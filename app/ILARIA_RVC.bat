@echo off
setlocal
set "ILARIA_RVC_HOME=%~dp0"
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; $root=$env:ILARIA_RVC_HOME; $python=Join-Path $root 'runtime\python.exe'; $launcher='ilaria\ilaria_launcher.py'; if (Test-Path -LiteralPath (Join-Path $root 'ilaria_launcher.py')) { $launcher='ilaria_launcher.py' }; if (!(Test-Path -LiteralPath $python) -or !(Test-Path -LiteralPath (Join-Path $root $launcher))) { throw 'Ilaria RVC runtime or launcher is missing.' }; Start-Process -FilePath $python -WorkingDirectory $root -ArgumentList @($launcher,'--pycmd','runtime\python.exe','--port','6969') -WindowStyle Hidden"
if errorlevel 1 pause
