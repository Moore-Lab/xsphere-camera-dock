@echo off
REM ===================================================================
REM  Launch the xsphere camera dock (standalone eval server).
REM  Double-click this (or the "xSphere Camera Dock" desktop shortcut).
REM  Edit the CAMERAS line to change which cameras the dock opens
REM  (tokens: zelux, zelux:SERIAL, sim). Ctrl+C in this window to stop.
REM ===================================================================
title xsphere camera dock
cd /d "%~dp0"

set "CAMERAS=zelux"
set "PORT=8000"

REM --- find a Python with the project's dependencies (conda base) ---
set "PY="
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Could not find Python. Install/activate the environment ^(conda base^) first.
  pause
  exit /b 1
)

echo Launching camera dock: %CAMERAS%
echo Open http://localhost:%PORT%/    ^(Ctrl+C in this window to stop^)
echo.

REM --- open the dashboard in the default browser once the server is up ---
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 4; Start-Process 'http://localhost:%PORT%/'"

"%PY%" -m camera_dock.webapp %CAMERAS% --host 0.0.0.0 --port %PORT%

echo.
echo Camera dock stopped. Press any key to close this window.
pause >nul
