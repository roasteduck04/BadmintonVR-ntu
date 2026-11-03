@echo off
rem BadmintonVR pipeline server (v0.8.2). Double-click to start; the browser opens http://localhost:8770.
rem Close this window (or press Ctrl+C) to stop the server.
rem
rem   start-pipeline-server.bat          Python on Windows, this folder = repository + Unity project (the laptop)
rem   start-pipeline-server.bat wsl      Python in WSL with the GPU venv (lab / remote GPU PC); this folder = the
rem                                      Unity project, the code is the WSL clone below (edit the WSL_ lines once)
rem   A plain double-click picks WSL by itself when this folder has no code (tools\pipeline_server\server.py), is
rem   inside WSL (\\wsl.localhost\...) or Windows has no Python. In WSL mode the Unity project is WSL_UNITY when that
rem   folder exists (the lab's), else this folder - so the file works from any folder on the lab PC.
rem
rem Guide: docs\setup\pipeline-server.md
setlocal

if not defined PORT set PORT=8770
set WSL_DISTRO=Ubuntu
set WSL_REPO=~/BadmintonVR
set WSL_VENV=~/venv-pc
set "WSL_UNITY=%USERPROFILE%\BadmintonVR-unity"

set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "UNITY=%HERE%"
if exist "%WSL_UNITY%\Assets\" set "UNITY=%WSL_UNITY%"

if not defined NO_BROWSER start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:%PORT%"

if /i "%~1"=="wsl" goto wsl
if "%HERE:~0,2%"=="\\" goto wsl
cd /d "%~dp0"
if not exist "tools\pipeline_server\server.py" goto wsl

rem a real Python with the server's packages (Windows' Microsoft Store stand-in python.exe passes `where python`)
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  where wsl >nul 2>&1
  if not errorlevel 1 goto wsl
  echo Python with the server packages was not found. Install Python 3.12 and: pip install -r tools\requirements.txt
  pause
  exit /b 1
)
echo Starting the pipeline server on Windows (port %PORT%) ...
python tools\pipeline_server\server.py --port %PORT%
goto done

:wsl
echo Starting the pipeline server in WSL (%WSL_DISTRO%, %WSL_REPO%, venv %WSL_VENV%), Unity project %UNITY% ...
if defined DRY_RUN goto dry
wsl -d %WSL_DISTRO% -- bash -lc "cd %WSL_REPO% && source %WSL_VENV%/bin/activate && python tools/pipeline_server/server.py --port %PORT% --unity-project \"$(wslpath '%UNITY%')\""

goto done

:dry
echo [dry run] wsl -d %WSL_DISTRO% -- bash -lc "cd %WSL_REPO% && source %WSL_VENV%/bin/activate && python tools/pipeline_server/server.py --port %PORT% --unity-project \"$(wslpath '%UNITY%')\""

:done
if errorlevel 1 (
  echo.
  echo The server stopped with an error ^(see above^).
  pause
)
endlocal
