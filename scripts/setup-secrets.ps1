# LiveTranslate Secrets Management Script
# Purpose: Set up secure environment variables and secrets for Docker containers
# Usage: .\scripts\setup-secrets.ps1 [-Environment "development|production"]

param(
    [string]$Environment = "development"
)

# Define secrets directory
$SecretsDir = ".secrets"
$EnvFile = ".env"

Write-Host "Setting up LiveTranslate secrets management..." -ForegroundColor Blue
Write-Host "Environment: $Environment" -ForegroundColor Cyan

# Create secrets directory if it doesn't exist
if (-not (Test-Path $SecretsDir)) {
    New-Item -ItemType Directory -Path $SecretsDir -Force | Out-Null
    Write-Host "Created secrets directory: $SecretsDir" -ForegroundColor Green
}

# Function to generate secure random password
function Generate-SecurePassword {
    param([int]$Length = 32)

    $chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    $password = ""
    for ($i = 0; $i -lt $Length; $i++) {
        $password += $chars[(Get-Random -Maximum $chars.Length)]
    }
    return $password
}

# Function to create secret file
function Create-SecretFile {
    param(
        [string]$SecretName,
        [string]$SecretValue,
        [string]$Description
    )

    $SecretFile = Join-Path $SecretsDir "$SecretName.txt"
    $SecretValue | Out-File -FilePath $SecretFile -Encoding UTF8 -NoNewline

    # Set restrictive permissions (Windows)
    try {
        $acl = Get-Acl $SecretFile
        $acl.SetAccessRuleProtection($true, $false)  # Disable inheritance
        $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            [System.Security.Principal.WindowsIdentity]::GetCurrent().Name,
            "FullControl",
            "Allow"
        )
        $acl.SetAccessRule($accessRule)
        Set-Acl -Path $SecretFile -AclObject $acl
        Write-Host "  Created secret: $SecretName ($Description)" -ForegroundColor Green
    }
    catch {
        Write-Host "  Created secret: $SecretName (permissions may need manual adjustment)" -ForegroundColor Yellow
    }
}

# Generate secrets based on environment
if ($Environment -eq "development") {
    Write-Host "`nGenerating development secrets..." -ForegroundColor Yellow

    # Development secrets (predictable for testing)
    Create-SecretFile -SecretName "jwt_secret" -SecretValue "dev-jwt-secret-key-change-in-production" -Description "JWT signing key"
    Create-SecretFile -SecretName "db_password" -SecretValue "dev-postgres-password" -Description "PostgreSQL password"
    Create-SecretFile -SecretName "redis_password" -SecretValue "dev-redis-password" -Description "Redis password"
    Create-SecretFile -SecretName "api_key_openai" -SecretValue "sk-dev-openai-key-placeholder" -Description "OpenAI API key"
    Create-SecretFile -SecretName "api_key_anthropic" -SecretValue "sk-ant-dev-anthropic-key-placeholder" -Description "Anthropic API key"
    Create-SecretFile -SecretName "api_key_google" -SecretValue "dev-google-api-key-placeholder" -Description "Google API key"
    Create-SecretFile -SecretName "session_secret" -SecretValue "dev-session-secret-key" -Description "Session encryption key"

} else {
    Write-Host "`nGenerating production secrets..." -ForegroundColor Red

    # Production secrets (randomly generated)
    Create-SecretFile -SecretName "jwt_secret" -SecretValue (Generate-SecurePassword -Length 64) -Description "JWT signing key"
    Create-SecretFile -SecretName "db_password" -SecretValue (Generate-SecurePassword -Length 32) -Description "PostgreSQL password"
    Create-SecretFile -SecretName "redis_password" -SecretValue (Generate-SecurePassword -Length 32) -Description "Redis password"
    Create-SecretFile -SecretName "session_secret" -SecretValue (Generate-SecurePassword -Length 64) -Description "Session encryption key"

    # API keys need to be set manually in production
    Create-SecretFile -SecretName "api_key_openai" -SecretValue "REPLACE_WITH_REAL_OPENAI_KEY" -Description "OpenAI API key"
    Create-SecretFile -SecretName "api_key_anthropic" -SecretValue "REPLACE_WITH_REAL_ANTHROPIC_KEY" -Description "Anthropic API key"
    Create-SecretFile -SecretName "api_key_google" -SecretValue "REPLACE_WITH_REAL_GOOGLE_KEY" -Description "Google API key"

    Write-Host "`nIMPORTANT: Update API keys in .secrets/ directory with real values!" -ForegroundColor Red
}

# Create .env file with secret references
Write-Host "`nCreating environment configuration..." -ForegroundColor Blue

$DebugValue = if ($Environment -eq "development") { "true" } else { "false" }
$LogLevel = if ($Environment -eq "development") { "DEBUG" } else { "INFO" }
$FlaskEnv = if ($Environment -eq "development") { "development" } else { "production" }
$FlaskDebug = if ($Environment -eq "development") { "1" } else { "0" }

$EnvContent = @"
# LiveTranslate Environment Configuration
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
# Environment: $Environment

# === APPLICATION SETTINGS ===
ENVIRONMENT=$Environment
DEBUG=$DebugValue
LOG_LEVEL=$LogLevel

# === SERVICE PORTS ===
FRONTEND_PORT=3000
WHISPER_SERVICE_PORT=5001
SPEAKER_SERVICE_PORT=5002
TRANSLATION_SERVICE_PORT=5003
GATEWAY_SERVICE_PORT=5000

# === DATABASE SETTINGS ===
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=livetranslate
POSTGRES_USER=livetranslate
# POSTGRES_PASSWORD loaded from secret file

# === REDIS SETTINGS ===
REDIS_HOST=redis
REDIS_PORT=6379
# REDIS_PASSWORD loaded from secret file

# === SECURITY SETTINGS ===
# JWT_SECRET loaded from secret file
# SESSION_SECRET loaded from secret file
SESSION_TIMEOUT=3600
CORS_ORIGINS=http://localhost:3000;http://localhost:5000

# === AI SERVICE SETTINGS ===
# API keys loaded from secret files
OPENAI_MODEL=gpt-4
ANTHROPIC_MODEL=claude-3-sonnet-20240229
GOOGLE_MODEL=gemini-pro

# === DOCKER SETTINGS ===
COMPOSE_PROJECT_NAME=livetranslate
DOCKER_BUILDKIT=1

# === MONITORING SETTINGS ===
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
GRAFANA_ADMIN_USER=admin
# GRAFANA_ADMIN_PASSWORD loaded from secret file

# === DEVELOPMENT SETTINGS ===
FLASK_ENV=$FlaskEnv
FLASK_DEBUG=$FlaskDebug
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
"@

$EnvContent | Out-File -FilePath $EnvFile -Encoding UTF8

Write-Host "  Created environment file: $EnvFile" -ForegroundColor Green

# Create secrets loading script for bash/sh
$LoadSecretsScript = @"
#!/bin/bash
# Load secrets from files into environment variables
# Source this script in your containers or applications

if [ -f .secrets/jwt_secret.txt ]; then
    export JWT_SECRET=`$(cat .secrets/jwt_secret.txt)
fi

if [ -f .secrets/db_password.txt ]; then
    export POSTGRES_PASSWORD=`$(cat .secrets/db_password.txt)
fi

if [ -f .secrets/redis_password.txt ]; then
    export REDIS_PASSWORD=`$(cat .secrets/redis_password.txt)
fi

if [ -f .secrets/api_key_openai.txt ]; then
    export OPENAI_API_KEY=`$(cat .secrets/api_key_openai.txt)
fi

if [ -f .secrets/api_key_anthropic.txt ]; then
    export ANTHROPIC_API_KEY=`$(cat .secrets/api_key_anthropic.txt)
fi

if [ -f .secrets/api_key_google.txt ]; then
    export GOOGLE_API_KEY=`$(cat .secrets/api_key_google.txt)
fi

if [ -f .secrets/session_secret.txt ]; then
    export SESSION_SECRET=`$(cat .secrets/session_secret.txt)
fi

echo "Secrets loaded from .secrets/ directory"
"@

$LoadSecretsScript | Out-File -FilePath "scripts/load-secrets.sh" -Encoding UTF8

Write-Host "  Created secrets loading script: scripts/load-secrets.sh" -ForegroundColor Green

# Create .gitignore entries for secrets
$GitignoreEntries = @"

# LiveTranslate Secrets (DO NOT COMMIT)
.secrets/
*.key
*.pem
*.p12
.env.local
.env.production
"@

if (Test-Path ".gitignore") {
    $GitignoreContent = Get-Content ".gitignore" -Raw -ErrorAction SilentlyContinue
    if ($GitignoreContent -notlike "*LiveTranslate Secrets*") {
        $GitignoreEntries | Add-Content ".gitignore"
        Write-Host "  Updated .gitignore with secrets exclusions" -ForegroundColor Green
    }
} else {
    $GitignoreEntries | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "  Created .gitignore with secrets exclusions" -ForegroundColor Green
}

# Display security summary
Write-Host "`nSecurity Setup Summary:" -ForegroundColor Blue
Write-Host "  Secrets directory: $SecretsDir (restricted permissions)" -ForegroundColor Cyan
Write-Host "  Environment file: $EnvFile" -ForegroundColor Cyan
Write-Host "  Secrets loader: scripts/load-secrets.sh" -ForegroundColor Cyan
Write-Host "  Git exclusions: Updated .gitignore" -ForegroundColor Cyan

Write-Host "`nSecurity Reminders:" -ForegroundColor Yellow
Write-Host "  • Never commit .secrets/ directory to version control" -ForegroundColor Yellow
Write-Host "  • Rotate secrets regularly in production" -ForegroundColor Yellow
Write-Host "  • Use proper file permissions on secret files" -ForegroundColor Yellow
Write-Host "  • Monitor secret access and usage" -ForegroundColor Yellow

if ($Environment -eq "production") {
    Write-Host "`nPRODUCTION SETUP REQUIRED:" -ForegroundColor Red
    Write-Host "  1. Replace placeholder API keys in .secrets/ with real values" -ForegroundColor Red
    Write-Host "  2. Verify file permissions on secret files" -ForegroundColor Red
    Write-Host "  3. Set up secret rotation procedures" -ForegroundColor Red
    Write-Host "  4. Configure monitoring for secret access" -ForegroundColor Red
}

Write-Host "`nSecrets management setup completed!" -ForegroundColor Green
