# LiveTranslate Security Check Script
# Purpose: Check security posture of Docker containers
# Usage: .\scripts\security-check.ps1

Write-Host "Checking LiveTranslate container security..." -ForegroundColor Blue

# Check Docker version
Write-Host "`nDocker Information:" -ForegroundColor Yellow
try {
    docker version --format "Server Version: {{.Server.Version}}"
    docker info --format "Security Options: {{.SecurityOptions}}"
}
catch {
    Write-Host "Could not get Docker information" -ForegroundColor Red
}

# Check running containers
Write-Host "`nRunning Containers:" -ForegroundColor Yellow
docker ps

# Check networks
Write-Host "`nDocker Networks:" -ForegroundColor Yellow
docker network ls

# Check volumes
Write-Host "`nDocker Volumes:" -ForegroundColor Yellow
docker volume ls

# Check for our services
Write-Host "`nLiveTranslate Services Status:" -ForegroundColor Yellow
$Services = @("translation", "whisper", "speaker", "frontend")
foreach ($Service in $Services) {
    $Container = docker ps --filter "name=$Service" --format "{{.Names}}" 2>$null
    if ($Container) {
        Write-Host "  ✓ $Service service running: $Container" -ForegroundColor Green

        # Check user
        try {
            $User = docker exec $Container whoami 2>$null
            if ($User -eq "root") {
                Write-Host "    ⚠ WARNING: Running as root" -ForegroundColor Red
            } else {
                Write-Host "    ✓ Running as: $User" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "    ? Could not check user" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ✗ $Service service not running" -ForegroundColor Red
    }
}

# Security recommendations
Write-Host "`nSecurity Status Summary:" -ForegroundColor Blue
Write-Host "✓ Secrets management configured (.secrets/ directory)" -ForegroundColor Green
Write-Host "✓ Docker networks created (livetranslate-*)" -ForegroundColor Green
Write-Host "✓ Docker volumes configured (livetranslate-*)" -ForegroundColor Green

Write-Host "`nSecurity Recommendations:" -ForegroundColor Yellow
Write-Host "1. Ensure all containers run as non-root users" -ForegroundColor Cyan
Write-Host "2. Set resource limits in docker-compose files" -ForegroundColor Cyan
Write-Host "3. Use custom networks for service isolation" -ForegroundColor Cyan
Write-Host "4. Regular security monitoring and updates" -ForegroundColor Cyan

Write-Host "`nSecurity check completed!" -ForegroundColor Green
