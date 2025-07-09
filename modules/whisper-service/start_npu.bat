@echo off
echo ========================================
echo Starting Whisper Service with NPU
echo ========================================
echo.

REM Set environment variables for NPU
set OPENVINO_DEVICE=NPU
set WHISPER_DEFAULT_MODEL=whisper-base
set LOG_LEVEL=INFO

REM Set models directory
set WHISPER_MODELS_DIR=%~dp0models

echo Configuration:
echo   Device: NPU (Intel AI Boost)
echo   Model: whisper-base
echo   Models: %WHISPER_MODELS_DIR%
echo.

REM Start the service
python "%~dp0start_npu.py"

pause