# Service Overview

Quick reference for all 4 core services.

## Frontend Service (:5173)
- **Tech**: React 18 + TypeScript
- **Purpose**: User interface
- **Hardware**: Browser
- **Module**: `modules/frontend-service/`

## Orchestration Service (:3000)
- **Tech**: Python + FastAPI
- **Purpose**: Central coordinator
- **Hardware**: CPU-optimized
- **Module**: `modules/orchestration-service/`

## Whisper Service (:5001)
- **Tech**: Python + OpenVINO
- **Purpose**: Speech-to-text
- **Hardware**: NPU/GPU/CPU
- **Module**: `modules/whisper-service/`

## Translation Service (:5003)
- **Tech**: Python + Transformers
- **Purpose**: Translation
- **Hardware**: GPU/CPU
- **Module**: `modules/translation-service/`

See [Container Architecture](./README.md) for detailed breakdown.
