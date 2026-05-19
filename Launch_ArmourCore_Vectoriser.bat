@echo off
REM ====================================================================
REM ArmourCore CDS Vectoriser — Windows double-click launcher.
REM
REM 1. Switches to the script's own folder so relative imports work
REM    regardless of where the user launches from.
REM 2. Invokes the system Python on the cross-platform launcher.
REM 3. Pauses on failure so error text is visible.
REM ====================================================================

setlocal
cd /d "%~dp0"

REM Prefer the "py" launcher on Windows since it picks the right Python.
REM Fall back to plain "python" if "py" is unavailable.
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "Launch_ArmourCore_Vectoriser.py" %*
) else (
    python "Launch_ArmourCore_Vectoriser.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ====================================================================
    echo Launcher exited with error code %ERRORLEVEL%.
    echo If the GUI did not open, check that Python 3.10+ is installed and
    echo that PyQt6 is available:  pip install PyQt6
    echo ====================================================================
    pause
)
endlocal
