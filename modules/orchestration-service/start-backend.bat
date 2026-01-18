@echo off
echo 🚀 Starting LiveTranslate Orchestration Service with Poetry
echo ============================================================

cd /d "%~dp0"

echo 📦 Installing dependencies...
poetry install --only main

echo 🔧 Installing dev dependencies...
poetry install --with dev

echo 🎵 Installing audio dependencies (optional)...
poetry install --with audio 2>nul || echo ⚠️ Audio dependencies skipped

echo 📊 Installing monitoring dependencies (optional)...
poetry install --with monitoring 2>nul || echo ⚠️ Monitoring dependencies skipped

echo.
echo 🚀 Starting backend service...
echo Backend URL: http://localhost:3000
echo API Docs: http://localhost:3000/docs
echo.

poetry run python src/main.py
