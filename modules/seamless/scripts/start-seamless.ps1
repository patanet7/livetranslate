$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Join-Path $scriptDir '..'
Set-Location $rootDir

if (Test-Path 'requirements.txt') {
  Write-Host "[INFO] Installing Python dependencies..."
  pip install -r requirements.txt | Write-Host
}

if (-not $env:SEAMLESS_MODEL) { $env:SEAMLESS_MODEL = 'facebook/seamless-m4t-v2-large' }
if (-not $env:DEVICE) { $env:DEVICE = 'cpu' }

Write-Host "[INFO] Starting Seamless demo service on :5007"
python -m uvicorn src.server:app --host 0.0.0.0 --port 5007
