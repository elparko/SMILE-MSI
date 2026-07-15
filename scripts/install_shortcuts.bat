@echo off
REM ============================================================================
REM Put a clickable "SMILE MSI" on your Desktop and in the Start Menu (searchable,
REM pinnable to the taskbar). The shortcut launches the app LIVE from this repo via
REM run_app.bat, so it always runs your current code -- to update, just `git pull`
REM in this folder and relaunch. No rebuild, no frozen .exe.
REM
REM Run this ONCE:  double-click, or  scripts\install_shortcuts.bat
REM ============================================================================
setlocal
cd /d "%~dp0.."
set "TARGET=%CD%\run_app.bat"
set "WORKDIR=%CD%"
REM Copy the icon to a STABLE per-user location and point the shortcut THERE, not into the
REM repo. A `git pull` rewrites/bumps files in the repo (and can drop Explorer's icon cache),
REM which blanks a shortcut whose icon lives in the repo; an out-of-repo copy never gets
REM touched by pulling, so the icon sticks.
set "ICONDIR=%LOCALAPPDATA%\SMILE MSI"
set "ICON=%ICONDIR%\SMILE MSI.ico"
if not exist "%ICONDIR%" mkdir "%ICONDIR%"
copy /y "%CD%\scripts\SMILE MSI.ico" "%ICON%" >nul

if not exist "%TARGET%" (
  echo ERROR: run_app.bat not found next to the repo root -- run this from inside the repo.
  pause
  exit /b 1
)

REM Single-line PowerShell (no caret line-continuation) creates both shortcuts.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$W=New-Object -ComObject WScript.Shell; foreach($dir in @([Environment]::GetFolderPath('Desktop'), (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))){ $p=Join-Path $dir 'SMILE MSI.lnk'; $l=$W.CreateShortcut($p); $l.TargetPath='%TARGET%'; $l.WorkingDirectory='%WORKDIR%'; $l.IconLocation='%ICON%'; $l.Description='SMILE MSI - runs live from the repo'; $l.Save(); Write-Host ('Created: ' + $p) }" || goto :err

REM Nudge Explorer to refresh its icon cache so the icon appears right away.
ie4uinit.exe -show >nul 2>&1

echo.
echo Done. "SMILE MSI" is now on your Desktop and in the Start Menu.
echo   - Double-click it to launch (the FIRST launch builds .venv -- a minute or two).
echo   - To update later: `git pull` in this repo, then relaunch. The shortcut always
echo     runs the current code.
echo   - The icon now lives in "%ICONDIR%", so pulling the repo will not blank it.
echo   - If a git pull changes dependencies and the app errors on start, delete the
echo     .venv folder and relaunch once to rebuild it.
echo.
pause
goto :eof

:err
echo.
echo Could not create the shortcuts -- see the message above.
pause
exit /b 1
