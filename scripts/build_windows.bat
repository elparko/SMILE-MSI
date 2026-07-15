@echo off
REM ============================================================================
REM Build SMILE MSI as a clickable Windows app (local fallback for the GitHub
REM Actions release build). Produces  dist\SMILE MSI\SMILE MSI.exe  -- a one-folder
REM app you can double-click. Zip the whole "dist\SMILE MSI" folder to share it.
REM
REM Usage:  double-click this file, or run  scripts\build_windows.bat
REM
REM This window STAYS OPEN on failure (press a key to close) so you can read the
REM error, and the full pip install log is written to  pip_install.log  at the repo
REM root. Force a specific interpreter with, e.g.:   set PY=py -3.12
REM ============================================================================
setlocal
cd /d "%~dp0.."
set "LOG=%CD%\build_windows.log"
echo SMILE MSI Windows build - %DATE% %TIME% > "%LOG%"
del /q "%CD%\pip_install.log" >nul 2>&1

REM --- 1. Pick a Python that actually has prebuilt wheels for the heavy compiled
REM deps (llvmlite / numba / PySide6). CI builds on 3.12 and it works; a too-new
REM "python" on PATH (e.g. 3.14) has no llvmlite wheel yet, so pip falls back to
REM compiling it from source and dies without LLVM + MSVC ("the build keeps
REM failing after building a wheel" report). Prefer the CI-proven versions via the
REM py launcher; fall back to whatever "python" is if the launcher is absent.
if not defined PY (
  for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
  )
)
if not defined PY ( python -c "import sys" >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo.
  echo ERROR: No Python found. Install Python 3.12 from https://www.python.org/downloads/
  echo and re-run this script ^(tick "Add python.exe to PATH" during install^).
  echo No Python found >> "%LOG%"
  goto :err
)
echo === Using interpreter: %PY% ===
%PY% --version
%PY% --version >> "%LOG%" 2>&1

REM Microsoft Store Python produces a broken bundle: PyInstaller can't bundle its redirected
REM libffi/UCRT DLLs, so the built .exe dies at launch with
REM   "ImportError: DLL load failed while importing _ctypes".
REM Detect it (its home lives under ...\WindowsApps\...) and steer to python.org.
for /f "delims=" %%p in ('%PY% -c "import sys;print(sys.base_prefix)" 2^>^&1') do set "PYBASE=%%p"
echo Interpreter home: %PYBASE% >> "%LOG%"
echo %PYBASE% | findstr /i "WindowsApps" >nul && (
  echo.
  echo WARNING: This looks like a Microsoft Store Python:
  echo   %PYBASE%
  echo Store Python usually builds an .exe that fails at launch with
  echo   "ImportError: DLL load failed while importing _ctypes".
  echo Install Python 3.12 from https://www.python.org/downloads/ ^(NOT the Store^), tick
  echo "Add python.exe to PATH", then re-run this script.
  echo Microsoft Store Python detected: %PYBASE% >> "%LOG%"
  echo.
)

REM --- 2. Warn (do not block) if the chosen Python is outside the wheel-safe range.
for /f "tokens=2" %%i in ('%PY% --version 2^>^&1') do set "PYVER=%%i"
echo Interpreter version: %PYVER% >> "%LOG%"
echo %PYVER% | findstr /r "^3\.1[0-3]\." >nul || (
  echo.
  echo WARNING: Python %PYVER% may not have prebuilt wheels for llvmlite/numba/PySide6.
  echo          If the install below fails while building a wheel, install Python 3.12
  echo          from https://www.python.org/downloads/ and re-run this script -- it will
  echo          prefer 3.12 automatically.
  echo.
)

REM --- 3. Fresh build venv. A leftover partial .venv-build from a failed run has a
REM half-installed dependency set that makes the next install fail in confusing ways.
echo === Creating build venv (.venv-build) ===
if exist ".venv-build" (
  echo Removing stale .venv-build...
  rmdir /s /q ".venv-build"
  if exist ".venv-build" (
    echo.
    echo ERROR: Could not fully remove the old .venv-build -- a file in it is locked.
    echo        Close any SMILE MSI / python still running from it, close any Explorer
    echo        window browsing inside it, then re-run this script.
    echo        ^(Antivirus can also briefly hold a handle -- if nothing is open, just re-run.^)
    echo Could not remove stale .venv-build >> "%LOG%"
    goto :err
  )
)
%PY% -m venv .venv-build || goto :err
call ".venv-build\Scripts\activate.bat" || goto :err

REM --- 4. Install gui + build + embed. --prefer-binary keeps pip on a prebuilt wheel
REM instead of building a newer source release; --log captures the FULL transcript so
REM the real error survives even if you miss it on screen. Upgrading setuptools/wheel
REM helps any source build that is genuinely unavoidable.
echo === Installing (gui + build + embed) ===
python -m pip install --upgrade pip setuptools wheel || goto :err
python -m pip install --prefer-binary --log "%CD%\pip_install.log" -e ".[gui,build,embed]" || goto :err

REM --- 5. Freeze to a one-folder, double-clickable app.
echo === Freezing with PyInstaller ===
pyinstaller --noconfirm smile_msi.spec || goto :err

REM PyInstaller can exit 0 yet leave no exe -- most often Windows Defender / enterprise
REM AV quarantines the freshly-built .exe as a false positive. Verify it exists so we
REM never print "Done" over a build that silently produced nothing.
if not exist "dist\SMILE MSI\SMILE MSI.exe" (
  echo.
  echo ERROR: PyInstaller finished but "%CD%\dist\SMILE MSI\SMILE MSI.exe" is missing.
  echo Some antivirus tools quarantine freshly-built .exe files as a false positive
  echo -- check Windows Security ^> Protection history / quarantine, then re-run.
  echo PyInstaller exited 0 but the app .exe is missing >> "%LOG%"
  goto :err
)

REM --- 6. Best-effort: drop a double-clickable shortcut on the Desktop. Never fatal.
echo === Creating Desktop shortcut (best effort) ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "$e=(Resolve-Path 'dist\SMILE MSI\SMILE MSI.exe').Path; $w=New-Object -ComObject WScript.Shell; $l=$w.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'SMILE MSI.lnk')); $l.TargetPath=$e; $l.WorkingDirectory=(Split-Path $e); $l.Save()" && echo Shortcut 'SMILE MSI' created on your Desktop. || echo (Could not create a Desktop shortcut -- skipped; you can still launch the .exe from the path shown below.)

echo.
echo === DONE ===
echo Your app:   "%CD%\dist\SMILE MSI\SMILE MSI.exe"   ^(double-click to run^)
echo Share it:   zip the whole  "dist\SMILE MSI"  folder.
echo.
pause
goto :eof

:err
echo.
echo ============================================================
echo BUILD FAILED. Scroll up to read the error above (this window
echo stays open). If it failed while building a wheel, the full
echo pip log is here:
echo   "%CD%\pip_install.log"
echo Most common fix: install Python 3.12 from
echo https://www.python.org/downloads/ and re-run this script.
echo ============================================================
echo.
pause
exit /b 1
