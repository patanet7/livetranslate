# LiveTranslate Docker Volume Restore Script
# Purpose: Restore Docker volumes from backup files
# Usage: .\scripts\restore-volumes.ps1 -BackupDir ".\backups\2025-06-07_20-30-00" [-VolumeNames "volume1,volume2"]

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDir,
    [string]$VolumeNames = "",
    [switch]$Force = $false
)

function Restore-Volume {
    param(
        [string]$VolumeName,
        [string]$BackupPath,
        [bool]$ForceRestore = $false
    )

    $BackupFile = Join-Path $BackupPath "$VolumeName.tar.gz"

    if (-not (Test-Path $BackupFile)) {
        Write-Host "  ❌ Backup file not found: $BackupFile" -ForegroundColor Red
        return $false
    }

    Write-Host "🔄 Restoring volume: $VolumeName" -ForegroundColor Yellow

    try {
        # Check if volume exists
        $VolumeExists = docker volume ls --format "{{.Name}}" | Where-Object { $_ -eq $VolumeName }

        if ($VolumeExists -and -not $ForceRestore) {
            Write-Host "  ⚠️  Volume $VolumeName already exists. Use -Force to overwrite." -ForegroundColor Orange
            return $false
        }

        if ($VolumeExists -and $ForceRestore) {
            Write-Host "  🗑️  Removing existing volume: $VolumeName" -ForegroundColor Orange
            docker volume rm $VolumeName | Out-Null
        }

        # Create new volume
        Write-Host "  📦 Creating volume: $VolumeName" -ForegroundColor Cyan
        docker volume create $VolumeName | Out-Null

        # Restore data from backup
        Write-Host "  📥 Restoring data from backup..." -ForegroundColor Cyan
        $Command = "docker run --rm -v ${VolumeName}:/data -v ${BackupPath}:/backup alpine tar xzf /backup/$VolumeName.tar.gz -C /data"
        Invoke-Expression $Command

        Write-Host "  ✅ Volume restored successfully: $VolumeName" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "  ❌ Error restoring $VolumeName`: $_" -ForegroundColor Red
        return $false
    }
}

# Validate backup directory
if (-not (Test-Path $BackupDir)) {
    Write-Host "❌ Backup directory not found: $BackupDir" -ForegroundColor Red
    exit 1
}

# Check for manifest file
$ManifestFile = Join-Path $BackupDir "backup-manifest.json"
if (Test-Path $ManifestFile) {
    Write-Host "📋 Loading backup manifest..." -ForegroundColor Blue
    $Manifest = Get-Content $ManifestFile | ConvertFrom-Json
    Write-Host "  📅 Backup created: $($Manifest.timestamp)" -ForegroundColor Cyan
    Write-Host "  🐳 Docker version: $($Manifest.docker_version)" -ForegroundColor Cyan
    Write-Host "  📦 Available backups: $($Manifest.backup_files.Count)" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  No manifest file found. Proceeding with available backup files..." -ForegroundColor Orange
}

# Check if Docker is running
try {
    docker version | Out-Null
}
catch {
    Write-Host "❌ Docker is not running or not accessible" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔄 Starting LiveTranslate volume restore..." -ForegroundColor Blue
Write-Host "Backup source: $BackupDir" -ForegroundColor Blue

# Determine which volumes to restore
$VolumesToRestore = @()
if ($VolumeNames) {
    $VolumesToRestore = $VolumeNames -split ","
    Write-Host "🎯 Restoring specific volumes: $($VolumesToRestore -join ', ')" -ForegroundColor Yellow
} else {
    # Get all available backup files
    $BackupFiles = Get-ChildItem -Path $BackupDir -Filter "*.tar.gz"
    $VolumesToRestore = $BackupFiles | ForEach-Object { $_.BaseName }
    Write-Host "📦 Restoring all available volumes: $($VolumesToRestore -join ', ')" -ForegroundColor Yellow
}

if ($VolumesToRestore.Count -eq 0) {
    Write-Host "❌ No volumes to restore" -ForegroundColor Red
    exit 1
}

# Confirm restore operation
if (-not $Force) {
    Write-Host "`n⚠️  WARNING: This operation will restore volumes and may overwrite existing data!" -ForegroundColor Red
    $Confirmation = Read-Host "Do you want to continue? (y/N)"
    if ($Confirmation -ne "y" -and $Confirmation -ne "Y") {
        Write-Host "❌ Restore operation cancelled" -ForegroundColor Yellow
        exit 0
    }
}

# Restore volumes
$SuccessCount = 0
$FailCount = 0

foreach ($VolumeName in $VolumesToRestore) {
    $Success = Restore-Volume -VolumeName $VolumeName -BackupPath $BackupDir -ForceRestore $Force
    if ($Success) {
        $SuccessCount++
    } else {
        $FailCount++
    }
}

# Display restore summary
Write-Host "`n📊 Restore Summary:" -ForegroundColor Blue
Write-Host "  ✅ Successfully restored: $SuccessCount volumes" -ForegroundColor Green
Write-Host "  ❌ Failed to restore: $FailCount volumes" -ForegroundColor Red

if ($SuccessCount -gt 0) {
    Write-Host "`n✅ Volume restore completed!" -ForegroundColor Green
    Write-Host "🔍 Verify restored volumes with: docker volume ls | findstr livetranslate" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ No volumes were restored successfully" -ForegroundColor Red
    exit 1
}
