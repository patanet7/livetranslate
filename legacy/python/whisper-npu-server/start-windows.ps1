# Whisper NPU Server - Windows Deployment Script
# This script helps you start the Whisper NPU Server on Windows with Docker

param(
    [string]$Mode = "npu",
    [switch]$Frontend = $false,
    [switch]$Stop = $false,
    [switch]$Restart = $false,
    [switch]$Logs = $false,
    [switch]$Status = $false,
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"

# Color functions
function Write-Success { param($Message) Write-Host $Message -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host $Message -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host $Message -ForegroundColor Red }
function Write-Info { param($Message) Write-Host $Message -ForegroundColor Cyan }

# Help function
function Show-Help {
    Write-Host @"
🧠 Whisper NPU Server - Windows Deployment Script

USAGE:
    .\start-windows.ps1 [OPTIONS]

OPTIONS:
    -Mode <mode>     Server mode: 'npu', 'cpu', or 'fallback' (default: npu)
    -Frontend        Start with frontend service on port 8080
    -Stop            Stop all services
    -Restart         Restart all services
    -Logs            Show container logs
    -Status          Show service status
    -Help            Show this help message

EXAMPLES:
    # Start NPU server
    .\start-windows.ps1

    # Start with frontend
    .\start-windows.ps1 -Frontend

    # Start CPU fallback
    .\start-windows.ps1 -Mode cpu

    # Stop all services
    .\start-windows.ps1 -Stop

    # Show logs
    .\start-windows.ps1 -Logs

ACCESS POINTS:
    - Server API: http://localhost:8009
    - Frontend: http://localhost:8080 (with -Frontend)
    - Settings: http://localhost:8080/settings.html

"@ -ForegroundColor White
}

# Check if Docker is available
function Test-Docker {
    try {
        docker --version | Out-Null
        return $true
    }
    catch {
        Write-Error "Docker is not installed or not running. Please install Docker Desktop for Windows."
        return $false
    }
}

# Check if Docker Compose is available
function Test-DockerCompose {
    try {
        docker-compose --version | Out-Null
        return $true
    }
    catch {
        Write-Error "Docker Compose is not available. Please update Docker Desktop."
        return $false
    }
}

# Get service status
function Get-ServiceStatus {
    Write-Info "Checking service status..."
    
    $services = @("whisper-server-npu", "whisper-server-cpu", "whisper-server-fallback", "whisper-frontend")
    
    foreach ($service in $services) {
        $status = docker ps -a --filter "name=$service" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        if ($status -and $status -ne "NAMES STATUS PORTS") {
            Write-Host $status
        }
    }
}

# Show logs
function Show-Logs {
    Write-Info "Showing container logs..."
    
    $containers = docker ps --filter "name=whisper-server" --format "{{.Names}}"
    
    if ($containers) {
        foreach ($container in $containers) {
            Write-Info "=== Logs for $container ==="
            docker logs --tail=50 $container
            Write-Host ""
        }
    } else {
        Write-Warning "No running Whisper server containers found."
    }
}

# Stop services
function Stop-Services {
    Write-Info "Stopping all Whisper services..."
    
    try {
        # Stop with all profiles
        docker-compose -f docker-compose.npu.yml --profile cpu-fallback --profile fallback --profile frontend down
        Write-Success "Services stopped successfully."
    }
    catch {
        Write-Error "Failed to stop services: $_"
    }
}

# Start services
function Start-Services {
    param($Mode, $IncludeFrontend)
    
    # Ensure models directory exists
    $modelsDir = "$env:USERPROFILE\.whisper\models"
    if (!(Test-Path $modelsDir)) {
        Write-Info "Creating models directory: $modelsDir"
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }
    
    # Build the command
    $profiles = @()
    
    switch ($Mode.ToLower()) {
        "cpu" { 
            $profiles += "cpu-fallback"
            $serviceName = "whisper-npu-server-cpu"
        }
        "fallback" { 
            $profiles += "fallback"
            $serviceName = "whisper-npu-server-prebuilt"
        }
        default { 
            $serviceName = "whisper-npu-server"
        }
    }
    
    if ($IncludeFrontend) {
        $profiles += "frontend"
    }
    
    # Build docker-compose command
    $composeCmd = "docker-compose -f docker-compose.npu.yml"
    
    foreach ($profile in $profiles) {
        $composeCmd += " --profile $profile"
    }
    
    if ($profiles.Count -gt 0) {
        $composeCmd += " up -d"
    } else {
        $composeCmd += " up -d $serviceName"
    }
    
    Write-Info "Starting Whisper NPU Server..."
    Write-Info "Mode: $Mode"
    Write-Info "Frontend: $($IncludeFrontend ? 'Yes' : 'No')"
    Write-Info "Command: $composeCmd"
    
    try {
        Invoke-Expression $composeCmd
        
        # Wait a moment for services to start
        Start-Sleep -Seconds 5
        
        Write-Success "Services started successfully!"
        
        # Show access information
        Write-Host ""
        Write-Info "🌐 Access Points:"
        Write-Host "  • Server API: http://localhost:8009" -ForegroundColor White
        
        if ($IncludeFrontend) {
            Write-Host "  • Frontend: http://localhost:8080" -ForegroundColor White
            Write-Host "  • Settings: http://localhost:8080/settings.html" -ForegroundColor White
        }
        
        Write-Host ""
        Write-Info "📊 Service Status:"
        Get-ServiceStatus
        
        Write-Host ""
        Write-Info "💡 Useful Commands:"
        Write-Host "  • View logs: .\start-windows.ps1 -Logs" -ForegroundColor Gray
        Write-Host "  • Stop services: .\start-windows.ps1 -Stop" -ForegroundColor Gray
        Write-Host "  • Restart: .\start-windows.ps1 -Restart" -ForegroundColor Gray
        
    }
    catch {
        Write-Error "Failed to start services: $_"
        Write-Info "Checking for issues..."
        Get-ServiceStatus
    }
}

# Main execution
function Main {
    Write-Host "🧠 Whisper NPU Server - Windows Deployment" -ForegroundColor Magenta
    Write-Host "============================================" -ForegroundColor Magenta
    
    if ($Help) {
        Show-Help
        return
    }
    
    # Check prerequisites
    if (!(Test-Docker)) { return }
    if (!(Test-DockerCompose)) { return }
    
    # Handle different actions
    if ($Status) {
        Get-ServiceStatus
        return
    }
    
    if ($Logs) {
        Show-Logs
        return
    }
    
    if ($Stop) {
        Stop-Services
        return
    }
    
    if ($Restart) {
        Write-Info "Restarting services..."
        Stop-Services
        Start-Sleep -Seconds 3
        Start-Services -Mode $Mode -IncludeFrontend $Frontend
        return
    }
    
    # Default: Start services
    Start-Services -Mode $Mode -IncludeFrontend $Frontend
}

# Run main function
Main 