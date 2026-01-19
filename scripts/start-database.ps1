#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start LiveTranslate Database Services
.DESCRIPTION
    Launches PostgreSQL, Redis, and pgAdmin for the LiveTranslate system
.PARAMETER Mode
    The startup mode: 'dev', 'prod', or 'test'
.PARAMETER Clean
    Whether to clean existing data (volumes)
.PARAMETER ShowLogs
    Whether to show real-time logs after startup
.EXAMPLE
    ./start-database.ps1 -Mode dev
    ./start-database.ps1 -Mode prod -Clean
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('dev', 'prod', 'test')]
    [string]$Mode = 'dev',

    [switch]$Clean,
    [switch]$ShowLogs,
    [switch]$Help
)

# Show help
if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Detailed
    exit 0
}

# Set error handling
$ErrorActionPreference = "Stop"

# Colors for output
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $Reset)
    Write-Host "$Color$Message$Reset"
}

function Test-DockerInstalled {
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Test-DatabaseRunning {
    try {
        $result = docker exec livetranslate-postgres pg_isready -U livetranslate -d livetranslate 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Wait-ForDatabase {
    param([int]$MaxAttempts = 30, [int]$DelaySeconds = 2)

    Write-ColorOutput "🔄 Waiting for database to be ready..." $Yellow

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        if (Test-DatabaseRunning) {
            Write-ColorOutput "✅ Database is ready!" $Green
            return $true
        }

        Write-Host "   Attempt $i/$MaxAttempts - waiting..."
        Start-Sleep -Seconds $DelaySeconds
    }

    Write-ColorOutput "❌ Database failed to start within expected time" $Red
    return $false
}

function Show-DatabaseInfo {
    Write-ColorOutput "`n📊 Database Service Information" $Blue
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Get container status
    $containers = @('livetranslate-postgres', 'livetranslate-redis', 'livetranslate-pgadmin')

    foreach ($container in $containers) {
        try {
            $status = docker ps --filter "name=$container" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --no-trunc
            if ($status) {
                Write-Host $status
            } else {
                Write-ColorOutput "❌ $container: Not running" $Red
            }
        }
        catch {
            Write-ColorOutput "❌ $container: Error getting status" $Red
        }
    }

    Write-Host "`n🔗 Connection Information:"
    Write-Host "   PostgreSQL: localhost:5432"
    Write-Host "   Database: livetranslate"
    Write-Host "   Username: livetranslate"
    Write-Host "   Redis: localhost:6379"
    Write-Host "   pgAdmin: http://localhost:8080"
    Write-Host ""
}

function Show-UsefulCommands {
    Write-ColorOutput "`n🛠️  Useful Commands" $Blue
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "   Check logs:        docker-compose -f docker-compose.database.yml logs -f"
    Write-Host "   Connect to DB:     docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate"
    Write-Host "   Connect to Redis:  docker exec -it livetranslate-redis redis-cli"
    Write-Host "   Stop services:     docker-compose -f docker-compose.database.yml down"
    Write-Host "   Clean volumes:     docker-compose -f docker-compose.database.yml down -v"
    Write-Host "   Restart service:   docker-compose -f docker-compose.database.yml restart postgres"
    Write-Host ""
}

# Main execution
Write-ColorOutput "🚀 Starting LiveTranslate Database Services" $Green
Write-ColorOutput "Mode: $Mode" $Blue

# Check Docker installation
if (-not (Test-DockerInstalled)) {
    Write-ColorOutput "❌ Docker or Docker Compose not found. Please install Docker Desktop." $Red
    exit 1
}

# Navigate to project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-ColorOutput "📁 Working directory: $projectRoot" $Blue

# Create docker configuration directories
$dockerDirs = @('docker/postgres', 'docker/redis', 'docker/pgadmin')
foreach ($dir in $dockerDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-ColorOutput "📁 Created directory: $dir" $Yellow
    }
}

# Create PostgreSQL configuration
$postgresConf = @"
# PostgreSQL Configuration for LiveTranslate
listen_addresses = '*'
port = 5432
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 4
effective_io_concurrency = 2
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB

# Logging
log_statement = 'all'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Performance monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all
"@

$postgresConf | Out-File -FilePath "docker/postgres/postgresql.conf" -Encoding UTF8

# Create pg_hba.conf
$pgHbaConf = @"
# PostgreSQL Client Authentication Configuration
local   all             all                                     scram-sha-256
host    all             all             127.0.0.1/32            scram-sha-256
host    all             all             ::1/128                 scram-sha-256
host    all             all             172.20.0.0/16           scram-sha-256
host    all             all             0.0.0.0/0               scram-sha-256
"@

$pgHbaConf | Out-File -FilePath "docker/postgres/pg_hba.conf" -Encoding UTF8

# Create Redis configuration
$redisConf = @"
# Redis Configuration for LiveTranslate
bind 0.0.0.0
port 6379
protected-mode no
timeout 300
tcp-keepalive 60
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
"@

$redisConf | Out-File -FilePath "docker/redis/redis.conf" -Encoding UTF8

# Create pgAdmin servers configuration
$pgAdminServers = @"
{
    "Servers": {
        "1": {
            "Name": "LiveTranslate Local",
            "Group": "Servers",
            "Host": "postgres",
            "Port": 5432,
            "MaintenanceDB": "livetranslate",
            "Username": "livetranslate",
            "SSLMode": "prefer",
            "Comment": "LiveTranslate PostgreSQL Database"
        }
    }
}
"@

$pgAdminServers | Out-File -FilePath "docker/pgadmin/servers.json" -Encoding UTF8

# Set environment variables based on mode
$env:POSTGRES_PASSWORD = if ($Mode -eq 'prod') { "secure_production_password_change_me" } else { "livetranslate_dev_password" }
$env:PGADMIN_EMAIL = "admin@livetranslate.local"
$env:PGADMIN_PASSWORD = if ($Mode -eq 'prod') { "secure_admin_password" } else { "admin" }

# Clean volumes if requested
if ($Clean) {
    Write-ColorOutput "🧹 Cleaning existing data volumes..." $Yellow
    docker-compose -f docker-compose.database.yml down -v --remove-orphans 2>$null

    # Remove named volumes explicitly
    $volumes = @('livetranslate_postgres_data', 'livetranslate_redis_data', 'livetranslate_pgadmin_data')
    foreach ($volume in $volumes) {
        try {
            docker volume rm $volume 2>$null
            Write-ColorOutput "   Removed volume: $volume" $Yellow
        }
        catch {
            # Volume might not exist, ignore
        }
    }
}

# Start services
Write-ColorOutput "🐳 Starting Docker containers..." $Yellow
try {
    docker-compose -f docker-compose.database.yml up -d --remove-orphans

    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed to start services"
    }

    Write-ColorOutput "✅ Docker containers started successfully" $Green
} catch {
    Write-ColorOutput "❌ Failed to start Docker containers: $($_.Exception.Message)" $Red
    exit 1
}

# Wait for database to be ready
if (-not (Wait-ForDatabase)) {
    Write-ColorOutput "❌ Database startup failed" $Red
    Write-ColorOutput "💡 Try running with -Clean flag to reset data" $Yellow
    exit 1
}

# Verify services are healthy
Start-Sleep -Seconds 5

# Show service information
Show-DatabaseInfo
Show-UsefulCommands

Write-ColorOutput "🎉 Database services started successfully!" $Green
Write-ColorOutput "💡 Use pgAdmin at http://localhost:8080 to manage the database" $Blue

# Show logs if requested
if ($ShowLogs) {
    Write-ColorOutput "`n📋 Showing real-time logs (Ctrl+C to exit)..." $Blue
    docker-compose -f docker-compose.database.yml logs -f
}

Write-ColorOutput "`n✅ Database startup complete!" $Green
