# Poetry Commands for Windows

## Quick Start (Run these commands in order)

```cmd
# Navigate to orchestration service directory
cd modules/orchestration-service

# Install Poetry (if not installed)
pip install poetry

# Install core dependencies
poetry install --only main

# Install development dependencies
poetry install --with dev

# Try to install audio dependencies (may fail on Windows - that's OK)
poetry install --with audio

# Try to install monitoring dependencies
poetry install --with monitoring

# Start the backend service
poetry run python src/main.py
```

## Alternative: One-line installation

```cmd
# Install all dependencies (some may fail on Windows)
poetry install --with dev,audio,monitoring

# Start the service
poetry run python src/main.py
```

## If audio dependencies fail (common on Windows):

```cmd
# Install only essential dependencies
poetry install --only main,dev

# Start the service (will work without audio processing)
poetry run python src/main.py
```

## Service URLs:
- Backend API: http://localhost:3000
- API Documentation: http://localhost:3000/docs
- Health Check: http://localhost:3000/api/health

## Frontend (run in separate terminal):
```cmd
cd modules/frontend-service
pnpm dev
```
- Frontend: http://localhost:5173

## Useful Poetry Commands:
```cmd
poetry shell                    # Activate virtual environment
poetry show                     # List installed packages
poetry env info                 # Show environment info
poetry install --help           # Show install options
poetry run pytest              # Run tests
poetry run black src/          # Format code
```