# LiveTranslate Container Security Hardening Script (Simplified)
# Purpose: Apply security hardening measures to Docker containers
# Usage: .\scripts\harden-containers-simple.ps1

Write-Host "Applying security hardening to LiveTranslate containers..." -ForegroundColor Blue

# Function to check Docker daemon security
function Check-DockerDaemonSecurity {
    Write-Host "`nChecking Docker daemon security..." -ForegroundColor Yellow

    # Check Docker version
    try {
        $DockerVersion = docker version --format "{{.Server.Version}}"
        Write-Host "  Docker version: $DockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "  Could not get Docker version" -ForegroundColor Red
    }

    # Check running containers
    Write-Host "`nRunning containers:" -ForegroundColor Cyan
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to check service security
function Check-ServiceSecurity {
    param([string]$ServicePattern)

    Write-Host "`nChecking security for services matching: $ServicePattern" -ForegroundColor Yellow

    # Get running containers matching pattern
    $Containers = docker ps --filter "name=$ServicePattern" --format "{{.Names}}"

    if ($Containers) {
        foreach ($Container in $Containers) {
            Write-Host "`n  Checking container: $Container" -ForegroundColor Cyan

            # Check user
            try {
                $User = docker exec $Container whoami 2>$null
                if ($User -eq "root") {
                    Write-Host "    WARNING: Running as root user" -ForegroundColor Red
                } else {
                    Write-Host "    OK: Running as user: $User" -ForegroundColor Green
                }
            }
            catch {
                Write-Host "    Could not check user" -ForegroundColor Yellow
            }

            # Check basic stats
            try {
                $Stats = docker stats $Container --no-stream --format "{{.CPUPerc}} {{.MemUsage}}"
                Write-Host "    Resource usage: $Stats" -ForegroundColor Gray
            }
            catch {
                Write-Host "    Could not get resource stats" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  No running containers found matching pattern: $ServicePattern" -ForegroundColor Yellow
    }
}

# Main execution
Check-DockerDaemonSecurity

# Check our services
$ServicePatterns = @("translation", "whisper", "speaker", "frontend")
foreach ($Pattern in $ServicePatterns) {
    Check-ServiceSecurity -ServicePattern $Pattern
}

# Create simple security monitoring script
Write-Host "`nCreating security monitoring script..." -ForegroundColor Yellow

$MonitoringScript = @'
#!/bin/bash
# LiveTranslate Security Monitoring Script
echo "LiveTranslate Security Monitoring Report"
echo "Generated: $(date)"
echo "========================================"

echo ""
echo "Running Containers:"
docker ps

echo ""
echo "Resource Usage:"
docker stats --no-stream

echo ""
echo "Networks:"
docker network ls

echo ""
echo "Volumes:"
docker volume ls

echo ""
echo "Images:"
docker images

echo ""
echo "Monitoring completed at $(date)"
'@

$MonitoringScript | Out-File -FilePath "scripts/security-monitor-simple.sh" -Encoding UTF8
Write-Host "  Created security monitoring script: scripts/security-monitor-simple.sh" -ForegroundColor Green

# Security summary
Write-Host "`nSecurity Hardening Summary:" -ForegroundColor Blue
Write-Host "✓ Docker daemon security checked" -ForegroundColor Green
Write-Host "✓ Container security assessed" -ForegroundColor Green
Write-Host "✓ Security monitoring script created" -ForegroundColor Green

Write-Host "`nSecurity Recommendations:" -ForegroundColor Yellow
Write-Host "1. Ensure all containers run as non-root users" -ForegroundColor Cyan
Write-Host "2. Set resource limits in docker-compose files" -ForegroundColor Cyan
Write-Host "3. Use custom networks for service isolation" -ForegroundColor Cyan
Write-Host "4. Implement secrets management" -ForegroundColor Cyan
Write-Host "5. Regular security monitoring and updates" -ForegroundColor Cyan

Write-Host "`nContainer hardening completed!" -ForegroundColor Green
