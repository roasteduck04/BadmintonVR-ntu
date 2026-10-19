@echo off
rem BadmintonVR pipeline server (v0.8.2). Double-click to start; the browser opens http://localhost:8770.
rem Close this window (or press Ctrl+C) to stop the server.
rem
rem   start-pipeline-server.bat          Python on Windows, this folder = repository + Unity project (the laptop)
rem   start-pipeline-server.bat wsl      Python in WSL with the GPU venv (lab / remote GPU PC); this folder = the
rem                                      Unity project, the code is the WSL clone below (edit the WSL_ lines once)
rem
rem Guide: docs\setup\pipeline-server.md
setlocal

if not defined PORT set PORT=8770
set WSL_DISTRO=Ubuntu
set WSL_REPO=~/BadmintonVR-ntu
set WSL_VENV=~/venv-pc

cd /d "%~dp0"
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

if not defined NO_BROWSER start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:%PORT%"

if /i "%~1"=="wsl" goto wsl

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 and the packages: pip install -r tools\requirements.txt
  pause
  exit /b 1
)
echo Starting the pipeline server on Windows (port %PORT%) ...
python tools\pipeline_server\server.py --port %PORT%
goto done

:wsl
echo Starting the pipeline server in WSL (%WSL_DISTRO%, %WSL_REPO%, venv %WSL_VENV%), Unity project %HERE% ...
wsl -d %WSL_DISTRO% -- bash -lc "cd %WSL_REPO% && source %WSL_VENV%/bin/activate && python tools/pipeline_server/server.py --port %PORT% --unity-project \"$(wslpath '%HERE%')\""

:done
if errorlevel 1 (
  echo.
  echo The server stopped with an error ^(see above^).
  pause
)
endlocal
