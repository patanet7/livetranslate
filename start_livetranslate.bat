@echo off
echo Starting LiveTranslate System...

:: Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Docker is not running! Please start Docker Desktop and try again.
    exit /b 1
)

:: Pull the latest code if needed
echo Updating code...
git pull

:: Build and start the containers
echo Building and starting containers...
docker-compose up -d --build

:: Wait for services to start
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

:: Check if containers are running
docker ps | findstr "livetranslate"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Containers failed to start. Check docker logs.
    echo To view logs, run: docker-compose logs
    exit /b 1
)

:: Open the web interface in default browser
echo Opening web interface...
start http://localhost:8080

echo.
echo LiveTranslate is now running!
echo.
echo Transcription server: ws://localhost:8765
echo Translation server: ws://localhost:8010
echo Web interface: http://localhost:8080
echo.
echo Available commands:
echo   View Docker logs: docker-compose logs
echo   Follow logs: docker-compose logs -f
echo   Start CSV logger: python python/logging_client.py
echo   Start audio client: python python/audio_client.py
echo.

:: Ask user what they want to do
echo What would you like to do?
echo 1. Start CSV logger (logs transcriptions and translations to files)
echo 2. Start audio client (stream audio to the server)
echo 3. View Docker logs
echo 4. Keep services running (press any key to stop later)
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting CSV logger...
    cd python
    python logging_client.py
    cd ..
) else if "%choice%"=="2" (
    echo Starting audio client...
    cd python
    python audio_client.py
    cd ..
) else if "%choice%"=="3" (
    echo Showing Docker logs...
    docker-compose logs
    echo.
    echo Press any key to continue...
    pause >nul
) else (
    echo Services are running in the background.
    echo Press any key to stop all services...
    pause >nul
)

echo Stopping all services...
docker-compose down
echo All services stopped. 