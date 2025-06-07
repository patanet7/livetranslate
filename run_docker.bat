@echo off
echo Building and running LiveTranslate Whisper Server...

REM Check if Docker is running
docker info > nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Build and start the container
echo Building container...
docker-compose build

echo Starting container...
docker-compose up -d

echo.
echo Container is running!
echo - API endpoint: http://localhost:8009
echo - Models directory: %USERPROFILE%\.whisper\models
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down 