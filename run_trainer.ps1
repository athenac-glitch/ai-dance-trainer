# AI K-Pop Dance Trainer PowerShell Launcher
Set-Location $PSScriptRoot
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AI K-Pop Dance Trainer (BA5077 Edge AIoT)" -ForegroundColor Green
Write-Host "  Student: Yu-Ting Chang (113401532)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Launching GUI Launcher (SOURCE/launcher.py)..." -ForegroundColor Yellow

$directPy = "$env:LOCALAPPDATA\Python\pythoncore-3.15-64\python.exe"
if (Test-Path $directPy) {
    & $directPy SOURCE/launcher.py
} else {
    python SOURCE/launcher.py
}
