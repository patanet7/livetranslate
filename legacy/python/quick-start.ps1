# Quick Start Script for Meeting Demo
# Run this in PowerShell to get your working setup running

Write-Host "🚀 Starting LiveTranslate Legacy Setup..." -ForegroundColor Green

# Set current directory
Set-Location "$PSScriptRoot"

Write-Host "📍 Current directory: $(Get-Location)" -ForegroundColor Yellow

# Option 1: Start Whisper NPU Server directly (fastest)
Write-Host "`n🔥 OPTION 1 - Start Whisper NPU Server directly:" -ForegroundColor Cyan
Write-Host "cd whisper-npu-server" -ForegroundColor White
Write-Host "python start-native.py" -ForegroundColor White

# Option 2: Start with Docker (if Python environment has issues)
Write-Host "`n🐳 OPTION 2 - Start with Docker:" -ForegroundColor Cyan
Write-Host "cd whisper-npu-server" -ForegroundColor White
Write-Host "docker-compose up -d" -ForegroundColor White

# Option 3: Manual Python startup
Write-Host "`n🐍 OPTION 3 - Manual Python startup:" -ForegroundColor Cyan
Write-Host "cd whisper-npu-server" -ForegroundColor White
Write-Host "python server.py" -ForegroundColor White

Write-Host "`n📋 For Translation Service (separate terminal):" -ForegroundColor Cyan
Write-Host "cd .." -ForegroundColor White
Write-Host "python translation_server.py" -ForegroundColor White

Write-Host "`n🌐 Access Points:" -ForegroundColor Green
Write-Host "- Whisper Frontend: http://localhost:5000" -ForegroundColor White
Write-Host "- Whisper API: http://localhost:5000/transcribe" -ForegroundColor White
Write-Host "- Translation: ws://localhost:8010" -ForegroundColor White

Write-Host "`n⚡ FASTEST START FOR MEETING:" -ForegroundColor Red
Write-Host "1. cd whisper-npu-server" -ForegroundColor Yellow
Write-Host "2. python start-native.py" -ForegroundColor Yellow
Write-Host "3. Open http://localhost:5000 in browser" -ForegroundColor Yellow

# Auto-start if user wants
$response = Read-Host "`nDo you want me to auto-start the whisper server now? (y/n)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "🚀 Starting Whisper NPU Server..." -ForegroundColor Green
    Set-Location "whisper-npu-server"
    python start-native.py
}