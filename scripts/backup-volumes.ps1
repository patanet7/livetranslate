# LiveTranslate Docker Volume Backup Script
# Purpose: Backup critical Docker volumes for data protection
# Usage: .\scripts\backup-volumes.ps1 [backup-directory]

param(
    [string]$BackupDir = ".\backups\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
)

# Create backup directory
Write-Host "Creating backup directory: $BackupDir" -ForegroundColor Green
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

# Define critical volumes to backup
$CriticalVolumes = @(
    "livetranslate-postgres-data",
    "livetranslate-redis-data",
    "livetranslate-sessions",
    "livetranslate-models-whisper",
    "livetranslate-models-speaker",
    "livetranslate-models-translation"
)

# Define optional volumes (can be recreated)
$OptionalVolumes = @(
    "livetranslate-audio-uploads",
    "livetranslate-logs",
    "livetranslate-prometheus-data",
    "livetranslate-grafana-data"
)

function Backup-Volume {
    param(
        [string]$VolumeName,
        [string]$BackupPath,
        [bool]$Critical = $true
    )

    $BackupFile = Join-Path $BackupPath "$VolumeName.tar.gz"
    $Priority = if ($Critical) { "CRITICAL" } else { "OPTIONAL" }

    Write-Host "[$Priority] Backing up volume: $VolumeName" -ForegroundColor $(if ($Critical) { "Yellow" } else { "Cyan" })

    try {
        # Create backup using Alpine container
        $Command = "docker run --rm -v ${VolumeName}:/data -v ${BackupPath}:/backup alpine tar czf /backup/$VolumeName.tar.gz -C /data ."
        Invoke-Expression $Command

        if (Test-Path $BackupFile) {
            $Size = (Get-Item $BackupFile).Length / 1MB
            Write-Host "  ✅ Backup completed: $BackupFile ($([math]::Round($Size, 2)) MB)" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Backup failed: $BackupFile not created" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  ❌ Error backing up $VolumeName`: $_" -ForegroundColor Red
    }
}

# Check if Docker is running
try {
    docker version | Out-Null
}
catch {
    Write-Host "❌ Docker is not running or not accessible" -ForegroundColor Red
    exit 1
}

Write-Host "🔄 Starting LiveTranslate volume backup..." -ForegroundColor Blue
Write-Host "Backup location: $BackupDir" -ForegroundColor Blue

# Backup critical volumes
Write-Host "`n📦 Backing up CRITICAL volumes..." -ForegroundColor Yellow
foreach ($Volume in $CriticalVolumes) {
    # Check if volume exists
    $VolumeExists = docker volume ls --format "{{.Name}}" | Where-Object { $_ -eq $Volume }
    if ($VolumeExists) {
        Backup-Volume -VolumeName $Volume -BackupPath $BackupDir -Critical $true
    } else {
        Write-Host "  ⚠️  Volume not found: $Volume" -ForegroundColor Orange
    }
}

# Backup optional volumes
Write-Host "`n📦 Backing up OPTIONAL volumes..." -ForegroundColor Cyan
foreach ($Volume in $OptionalVolumes) {
    # Check if volume exists
    $VolumeExists = docker volume ls --format "{{.Name}}" | Where-Object { $_ -eq $Volume }
    if ($VolumeExists) {
        Backup-Volume -VolumeName $Volume -BackupPath $BackupDir -Critical $false
    } else {
        Write-Host "  ⚠️  Volume not found: $Volume" -ForegroundColor Orange
    }
}

# Create backup manifest
$ManifestFile = Join-Path $BackupDir "backup-manifest.json"
$Manifest = @{
    timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    backup_directory = $BackupDir
    critical_volumes = $CriticalVolumes
    optional_volumes = $OptionalVolumes
    docker_version = (docker version --format "{{.Server.Version}}")
    backup_files = @()
}

# Add backup file information to manifest
Get-ChildItem -Path $BackupDir -Filter "*.tar.gz" | ForEach-Object {
    $Manifest.backup_files += @{
        volume_name = $_.BaseName
        file_name = $_.Name
        file_size_mb = [math]::Round($_.Length / 1MB, 2)
        created = $_.CreationTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
}

$Manifest | ConvertTo-Json -Depth 3 | Out-File -FilePath $ManifestFile -Encoding UTF8

Write-Host "`n✅ Backup completed successfully!" -ForegroundColor Green
Write-Host "📁 Backup location: $BackupDir" -ForegroundColor Green
Write-Host "📋 Manifest file: $ManifestFile" -ForegroundColor Green

# Display backup summary
$TotalSize = (Get-ChildItem -Path $BackupDir -Filter "*.tar.gz" | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "📊 Total backup size: $([math]::Round($TotalSize, 2)) MB" -ForegroundColor Green
Write-Host "📦 Files backed up: $($Manifest.backup_files.Count)" -ForegroundColor Green
