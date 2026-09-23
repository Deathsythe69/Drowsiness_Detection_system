# Launch the Drowsiness Detection System GUI
param(
    [switch]$Test,
    [switch]$Check,
    [switch]$Help
)

Set-Location $PSScriptRoot

if ($Help) {
    Write-Host "Usage: .\run.ps1 [-Test] [-Check] [-Help]"
    Write-Host "  -Test   Run automated pytest test suite"
    Write-Host "  -Check  Verify installed dependencies"
    Write-Host "  -Help   Show this help message"
    exit 0
}

# Check virtual environment
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[INFO] Creating virtual environment in .venv..." -ForegroundColor Cyan
    python -m venv .venv
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
}

if ($Check) {
    Write-Host "[CHECK] Verifying installed dependencies..." -ForegroundColor Cyan
    & ".\.venv\Scripts\python.exe" -c "import cv2, mediapipe, PyQt6, yaml, matplotlib, psutil, flask, qrcode; print('[OK] All required packages installed.')"
    exit $LASTEXITCODE
}

if ($Test) {
    Write-Host "[TESTS] Running automated test suite..." -ForegroundColor Cyan
    & ".\.venv\Scripts\python.exe" -m pytest -v
    exit $LASTEXITCODE
}

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "      Drowsiness & Driver Attention Detection System (v2.6)         " -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "[STARTING] Launching GUI application..." -ForegroundColor Yellow

& ".\.venv\Scripts\python.exe" "main.py" @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Application exited with code $LASTEXITCODE" -ForegroundColor Red
}
