@echo off
REM Double-click launcher (Windows) for the SMILE MSI desktop app.
REM First run builds a local environment; afterwards it opens instantly.
REM NOTE: no parenthesized if-blocks here on purpose -- an unescaped ")" inside
REM an echo string closes the block early and cmd dies with "was unexpected at
REM this time." Use goto/labels instead.
cd /d "%~dp0"

REM Prefer the py launcher (handles multiple Python versions); fall back to python.
set "PY=py"
where py >nul 2>nul || set "PY=python"

if exist ".venv\Scripts\python.exe" goto run

echo First run: building the environment. This takes a minute or two...
%PY% -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[gui]"
if errorlevel 1 goto fail

:run
.venv\Scripts\python -m smile_msi.gui
if errorlevel 1 goto fail
goto :eof

:fail
echo.
echo *** Something went wrong -- read the messages above. ***
echo If you see "python was not recognized", install Python 3.12+ from python.org
echo and check "Add python.exe to PATH" during setup.
pause
exit /b 1
