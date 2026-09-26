@echo off
REM ============================================================
REM  ISDO - Fix "DLL load failed ... onnxruntime" on Windows
REM  1) Installs/updates the Microsoft Visual C++ Redistributable (x64)
REM  2) Reinstalls onnxruntime inside labenv
REM  3) Tests the import and runs labs\C1\kb_setup.py
REM  Double-click this file, or run it from the ISDO-Claude Folder.
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY=%~dp0labenv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Could not find labenv\Scripts\python.exe next to this file.
    goto :end
)

echo.
echo [1/3] Downloading Microsoft Visual C++ Redistributable (x64) from aka.ms ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile \"$env:TEMP\vc_redist.x64.exe\" -UseBasicParsing"
if errorlevel 1 (
    echo [ERROR] Download failed. Check your internet connection and try again.
    goto :end
)

echo      Installing - click YES on the Windows admin prompt when it appears ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath \"$env:TEMP\vc_redist.x64.exe\" -ArgumentList '/install','/passive','/norestart' -Verb RunAs -Wait -PassThru; exit $p.ExitCode"
REM 0 = installed, 1638 = newer version already present, 3010 = reboot needed
echo      Installer exit code: %errorlevel%  (0, 1638 or 3010 are all OK)

echo.
echo [2/3] Reinstalling onnxruntime in labenv ...
"%PY%" -m pip uninstall -y onnxruntime
"%PY%" -m pip install --no-cache-dir onnxruntime

echo.
echo [3/3] Testing onnxruntime import ...
"%PY%" -c "import onnxruntime; print('onnxruntime OK, version', onnxruntime.__version__)"
if errorlevel 1 (
    echo.
    echo [STILL FAILING] onnxruntime cannot load on this Python version.
    echo Next step: create a Python 3.12 environment - see the chat for commands.
    goto :end
)

echo.
echo Running labs\C1\kb_setup.py ...
echo ------------------------------------------------------------
"%PY%" "labs\C1\kb_setup.py"

:end
echo.
pause