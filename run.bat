@echo off
cd /d "%~dp0"
echo Starting Drowsiness Detection System...
call .venv\Scripts\activate.bat
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
