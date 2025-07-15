@echo off
echo Cleaning and restarting orchestration service...
echo ============================================

cd /d "%~dp0"

echo.
echo Step 1: Cleaning Python cache...
for /r %%i in (__pycache__) do if exist "%%i" rd /s /q "%%i"
for /r %%i in (*.pyc) do if exist "%%i" del /q "%%i"

echo.
echo Step 2: Killing any existing Python processes for orchestration...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *orchestration*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *main.py*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 3: Starting fresh orchestration service...
call start-backend.bat