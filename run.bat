@echo off
setlocal enabledelayedexpansion

:: Navigate to script directory
cd /d "%~dp0"

title Drowsiness ^& Attention Detection System

echo =====================================================================
echo       Drowsiness ^& Driver Attention Detection System (v2.6)
echo =====================================================================
echo.

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

:: 1. Check if virtual environment exists; if not, create it
if not exist "%PYTHON_EXE%" (
    echo [INFO] Virtual environment not found at .venv.
    echo [INFO] Searching for system Python installation...
    
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "SYS_PY=py -3"
    ) else (
        where python >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            set "SYS_PY=python"
        ) else (
            echo.
            echo [ERROR] Python 3.10+ is required but was not found on your PATH.
            echo Please install Python from https://www.python.org/downloads/
            echo and ensure "Add Python to PATH" is checked during installation.
            echo.
            pause
            exit /b 1
        )
    )

    echo [SETUP] Creating virtual environment in .venv using !SYS_PY!...
    !SYS_PY! -m venv "%VENV_DIR%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    
    echo [SETUP] Upgrading pip...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    
    if exist "requirements.txt" (
        echo [SETUP] Installing required dependencies from requirements.txt...
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        if !ERRORLEVEL! NEQ 0 (
            echo [ERROR] Dependency installation encountered errors.
            pause
            exit /b 1
        )
    )
)

:: 2. Activate virtual environment if activation script exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
)

:: 3. Argument handlers: --help, test / --test, --check
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="test" goto run_tests
if /i "%~1"=="--test" goto run_tests
if /i "%~1"=="--check" goto check_deps

:: 4. Fast integrity check for core packages
"%PYTHON_EXE%" -c "import cv2, mediapipe, PyQt6, yaml" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Missing core dependencies detected.
    echo [SETUP] Installing dependencies from requirements.txt...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install required dependencies.
        pause
        exit /b 1
    )
)

:: 5. Launch GUI Application
echo [STARTING] Launching GUI application...
echo.
"%PYTHON_EXE%" main.py %*
set "EXIT_CODE=!ERRORLEVEL!"

if !EXIT_CODE! NEQ 0 (
    echo.
    echo =====================================================================
    echo [ERROR] Application terminated with exit code !EXIT_CODE!.
    if exist "crash.log" (
        echo Check 'crash.log' in the project directory for details.
    )
    echo =====================================================================
    echo.
    pause
    exit /b !EXIT_CODE!
) else (
    echo.
    echo [INFO] Application exited normally.
)
exit /b 0

:run_tests
echo [TESTS] Running automated test suite (pytest)...
echo.
"%PYTHON_EXE%" -m pytest -v
set "TEST_CODE=!ERRORLEVEL!"
echo.
pause
exit /b !TEST_CODE!

:check_deps
echo [CHECK] Verifying installed dependencies...
"%PYTHON_EXE%" -c "import cv2, mediapipe, PyQt6, yaml, matplotlib, psutil, flask, qrcode; print('[OK] All required packages are installed and functional.')"
exit /b !ERRORLEVEL!

:show_help
echo Usage: run.bat [options]
echo.
echo Options:
echo   (no arguments)     Launch the Drowsiness Detection System GUI
echo   test, --test       Run the automated pytest test suite
echo   --check            Verify all required dependencies
echo   -h, --help         Display this help message
echo.
exit /b 0
