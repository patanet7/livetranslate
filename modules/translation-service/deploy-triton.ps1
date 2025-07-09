# PowerShell script to deploy Triton Translation Service
# Run this from the translation-service directory

Write-Host "Deploying Triton Translation Service..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "Dockerfile.triton")) {
    Write-Host "Error: Please run this script from modules/translation-service directory" -ForegroundColor Red
    exit 1
}

# Go to project root for build context
Write-Host "Changing to project root for build context..." -ForegroundColor Yellow
cd ../../

# Verify we're in project root
if (-not (Test-Path "modules/shared/src/inference")) {
    Write-Host "Error: Cannot find shared inference modules" -ForegroundColor Red
    exit 1
}

Write-Host "Building and starting Triton Translation Service..." -ForegroundColor Green

# Use the simple compose file with correct build context
docker-compose -f modules/translation-service/docker-compose-triton-simple.yml up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Services available at:" -ForegroundColor Cyan
    Write-Host "  Triton Server: http://localhost:8000" -ForegroundColor White
    Write-Host "  Translation API: http://localhost:5003" -ForegroundColor White
    Write-Host "  Metrics: http://localhost:8002/metrics" -ForegroundColor White
    Write-Host ""
    Write-Host "Test with:" -ForegroundColor Cyan
    Write-Host "  python modules/translation-service/test_triton_simple.py" -ForegroundColor White
} else {
    Write-Host "Deployment failed!" -ForegroundColor Red
    exit 1
}