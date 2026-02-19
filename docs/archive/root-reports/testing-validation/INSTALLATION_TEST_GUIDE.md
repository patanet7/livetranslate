# Dependency Standardization - Installation & Testing Guide
**Date:** 2026-01-05
**Standardized Versions:** pytest 8.4.2, numpy 2.3.4, fastapi 0.121.0, redis 6.4.0, and 40+ more

---

## Quick Start

```bash
# 1. Validate dependencies can install
./scripts/validate_dependencies.sh

# 2. Test each service (in order)
cd modules/whisper-service && pip install -r requirements.txt && pytest tests/ -v
cd modules/orchestration-service && pip install -r requirements.txt && pytest tests/ -v
cd modules/translation-service && pip install -r requirements-cpu.txt && pytest tests/ -v
cd modules/bot-container && pip install -r requirements.txt && pytest tests/ -v
```

---

## Detailed Installation Instructions

### Prerequisites

```bash
# Ensure you have Python 3.10+ (3.12+ recommended for orchestration)
python3 --version

# Ensure pip is up to date
python3 -m pip install --upgrade pip

# Recommended: Use virtual environments
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

---

## Service-by-Service Installation

### 1. Whisper Service (NPU/GPU Optimized)

**Location:** `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service`

#### Step 1: Install Core Dependencies
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service

# Install main dependencies
pip install -r requirements.txt

# Expected output:
# ✓ numpy==2.3.4
# ✓ scipy==1.16.2
# ✓ librosa==0.11.0
# ✓ soundfile==0.13.1
# ✓ websockets==15.0.1
# ✓ redis==6.4.0
# ✓ fastapi==0.121.0
# ... and 40+ more packages
```

#### Step 2: Install Test Dependencies
```bash
# Install test requirements
pip install -r tests/requirements-test.txt

# Expected output:
# ✓ pytest==8.4.2
# ✓ pytest-asyncio==1.2.0
# ✓ pytest-cov==7.0.0
# ✓ pytest-mock==3.15.1
# ✓ pytest-xdist==3.8.0
# ✓ hypothesis==6.145.1
```

#### Step 3: Verify Installation
```bash
# Verify critical packages
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# Expected: NumPy: 2.3.4

python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')"
# Expected: SciPy: 1.16.2

python3 -c "import librosa; print(f'librosa: {librosa.__version__}')"
# Expected: librosa: 0.11.0

python3 -c "import pytest; print(f'pytest: {pytest.__version__}')"
# Expected: pytest: 8.4.2

python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
# Expected: FastAPI: 0.121.0
```

#### Step 4: Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v -m "not slow"
pytest tests/integration/ -v -m integration
```

#### Step 5: Test Audio Processing (CRITICAL)
```bash
# Test NumPy 2.x compatibility with audio pipeline
pytest tests/unit/test_audio_processing.py -v

# Test real audio files (if available)
pytest tests/integration/test_whisper_service.py -v

# Expected: All tests should PASS
# If failures occur, check NumPy 2.x compatibility issues
```

---

### 2. Orchestration Service (Backend API)

**Location:** `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service`

#### Step 1: Install Core Dependencies
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service

# Install main dependencies
pip install -r requirements.txt

# Expected output:
# ✓ fastapi==0.121.0
# ✓ uvicorn==0.38.0
# ✓ pydantic==2.12.3
# ✓ websockets==15.0.1
# ✓ numpy==2.3.4
# ✓ scipy==1.16.2
```

#### Step 2: Install Optional Dependencies
```bash
# Database support
pip install -r requirements-database.txt
# ✓ asyncpg==0.30.0
# ✓ aiofiles==24.1.0

# Google Meet bot support
pip install -r requirements-google-meet.txt
# ✓ httpx==0.28.1
# ✓ google-api-python-client>=2.0.0

# Integration test dependencies
pip install -r tests/integration/requirements.txt
# ✓ pytest==8.4.2
# ✓ pytest-asyncio==1.2.0
# ✓ pytest-cov==7.0.0
# ✓ pytest-mock==3.15.1
```

#### Step 3: Verify Installation (Poetry)
```bash
# If using Poetry
poetry install
poetry install --with dev --with test --with monitoring

# Verify
poetry show | grep pytest
# pytest                    8.4.2

poetry show | grep fastapi
# fastapi                   0.121.0

poetry show | grep numpy
# numpy                     2.3.4
```

#### Step 4: Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with markers
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
```

#### Step 5: Test API Endpoints
```bash
# Start the service
uvicorn src.orchestration_service:app --host 0.0.0.0 --port 3000

# In another terminal, test endpoints
curl http://localhost:3000/api/health
# Expected: {"status": "healthy", "version": "1.0.0"}

curl http://localhost:3000/api/settings
# Expected: JSON configuration
```

---

### 3. Translation Service (GPU Optimized)

**Location:** `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/translation-service`

#### Step 1: Install Core Dependencies (CPU Version)
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/translation-service

# CPU-only installation (no GPU libraries)
pip install -r requirements-cpu.txt

# Expected output:
# ✓ fastapi==0.121.0
# ✓ uvicorn==0.38.0
# ✓ redis==6.4.0
# ✓ httpx==0.28.1
# ✓ websockets==15.0.1
# ✓ numpy==2.3.4
# ✓ python-socketio==5.14.2
```

#### Step 2: Install Full Dependencies (GPU Version - Optional)
```bash
# Full installation with vLLM and GPU support
pip install -r requirements.txt

# Note: This includes large ML packages:
# ✓ vllm>=0.6.0
# ✓ transformers>=4.40.0
# ✓ torch>=2.7.0 (large download)
```

#### Step 3: Verify Installation
```bash
# Verify critical packages
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
# Expected: FastAPI: 0.121.0

python3 -c "import redis; print(f'Redis: {redis.__version__}')"
# Expected: Redis: 6.4.0

python3 -c "import httpx; print(f'HTTPX: {httpx.__version__}')"
# Expected: HTTPX: 0.28.1

python3 -c "import websockets; print(f'WebSockets: {websockets.__version__}')"
# Expected: WebSockets: 15.0.1
```

#### Step 4: Run Tests
```bash
# Run tests (with pytest 8.4.2)
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

### 4. Bot Container

**Location:** `/Users/thomaspatane/Documents/GitHub/livetranslate/modules/bot-container`

#### Step 1: Install Dependencies
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/bot-container

# Install all dependencies
pip install -r requirements.txt

# Expected output:
# ✓ websockets==15.0.1
# ✓ httpx==0.28.1
# ✓ redis==6.4.0
# ✓ numpy==2.3.4
# ✓ scipy==1.16.2
# ✓ opencv-python==4.12.0.88
# ✓ pillow==12.0.0
# ✓ pytest==8.4.2
# ✓ pytest-asyncio==1.2.0
# ✓ pytest-cov==7.0.0
# ✓ pytest-mock==3.15.1
```

#### Step 2: Verify Installation
```bash
# Verify video processing libraries
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
# Expected: OpenCV: 4.12.0

python3 -c "import PIL; print(f'Pillow: {PIL.__version__}')"
# Expected: Pillow: 12.0.0

python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
# Expected: NumPy: 2.3.4
```

#### Step 3: Run Tests
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

### 5. Integration Tests

**Location:** `/Users/thomaspatane/Documents/GitHub/livetranslate/tests/integration`

#### Step 1: Install Test Dependencies
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/tests/integration

# Install comprehensive test requirements
pip install -r requirements-test.txt

# Expected output:
# ✓ pytest==8.4.2
# ✓ pytest-asyncio==1.2.0
# ✓ pytest-cov==7.0.0
# ✓ pytest-mock==3.15.1
# ✓ pytest-timeout==2.4.0
# ✓ pytest-xdist==3.8.0
# ✓ hypothesis==6.145.1
# ✓ faker==37.11.0
# ✓ fakeredis==2.32.0
# ✓ sqlalchemy==2.0.44
# ✓ numpy==2.3.4
# ✓ scipy==1.16.2
# ✓ httpx==0.28.1
# ✓ websockets==15.0.1
```

#### Step 2: Run Integration Tests
```bash
# Run all integration tests
pytest . -v

# Run with parallel execution
pytest . -v -n auto

# Run with coverage
pytest . -v --cov=../modules --cov-report=html
```

---

## Common Installation Issues & Solutions

### Issue 1: NumPy 2.x Compatibility Errors

**Symptom:**
```
AttributeError: module 'numpy' has no attribute 'int'
DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`
```

**Solution:**
```bash
# Check NumPy version
python3 -c "import numpy; print(numpy.__version__)"

# If < 2.3.4, upgrade
pip install --upgrade "numpy>=2.3.4"

# Update code using deprecated aliases:
# np.int → int
# np.float → float
# np.bool → bool
```

### Issue 2: pytest-cov 7.0.0 Coverage Differences

**Symptom:**
```
Coverage percentages differ from previous runs
Different files included in coverage report
```

**Solution:**
```bash
# Clear previous coverage data
rm -rf .coverage htmlcov/

# Run with new coverage engine
pytest tests/ -v --cov=src --cov-report=html

# Review coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Issue 3: Pydantic 2.12.3 Validation Errors

**Symptom:**
```
ValidationError: Input should be a valid string
Field required [type=missing, input_value=...]
```

**Solution:**
```bash
# Pydantic 2.x is stricter - update models:
# Old: field: str = None
# New: field: Optional[str] = None

# Or use ConfigDict for backward compatibility:
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(extra='forbid')  # or 'allow'
```

### Issue 4: FastAPI 0.121.0 + Pydantic 2.12.3 Schema Issues

**Symptom:**
```
OpenAPI schema generation errors
Request validation failures
```

**Solution:**
```bash
# Check FastAPI compatibility
python3 -c "import fastapi; import pydantic; print(f'FastAPI {fastapi.__version__}, Pydantic {pydantic.__version__}')"

# Should output: FastAPI 0.121.0, Pydantic 2.12.3

# Update response models:
from pydantic import BaseModel, Field

class Response(BaseModel):
    status: str = Field(..., description="Status message")
```

### Issue 5: Redis 6.4.0 Connection Issues

**Symptom:**
```
redis.exceptions.ConnectionError
AttributeError: 'Redis' object has no attribute '...'
```

**Solution:**
```bash
# Check Redis version
python3 -c "import redis; print(redis.__version__)"

# If < 6.4.0, upgrade
pip install --upgrade "redis>=6.4.0"

# Update async code to use proper API:
import redis.asyncio as redis
client = redis.Redis(host='localhost', port=6379, decode_responses=True)
```

---

## Performance Testing

### Test 1: NumPy 2.x Performance
```bash
# Benchmark NumPy operations
python3 << 'EOF'
import numpy as np
import time

# Create large array
arr = np.random.rand(10000, 10000)

start = time.time()
result = np.dot(arr, arr.T)
elapsed = time.time() - start

print(f"NumPy 2.x matrix multiply: {elapsed:.3f}s")
# Expected: <5 seconds (with optimizations)
EOF
```

### Test 2: FastAPI 0.121.0 Performance
```bash
# Benchmark request handling (requires service running)
pip install httpx

python3 << 'EOF'
import httpx
import asyncio
import time

async def benchmark():
    async with httpx.AsyncClient() as client:
        start = time.time()
        tasks = [client.get("http://localhost:3000/api/health") for _ in range(1000)]
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"1000 requests in {elapsed:.3f}s ({1000/elapsed:.1f} req/s)")

asyncio.run(benchmark())
EOF
```

### Test 3: Redis 6.4.0 Performance
```bash
# Benchmark Redis operations
python3 << 'EOF'
import redis
import time

client = redis.Redis(host='localhost', port=6379, decode_responses=True)

start = time.time()
for i in range(10000):
    client.set(f"key:{i}", f"value:{i}")
    client.get(f"key:{i}")
elapsed = time.time() - start

print(f"20,000 Redis ops in {elapsed:.3f}s ({20000/elapsed:.1f} ops/s)")
EOF
```

---

## Verification Checklist

### Installation Verification
- [ ] `./scripts/validate_dependencies.sh` passes
- [ ] All services install without errors
- [ ] `pytest --version` shows 8.4.2
- [ ] `python3 -c "import numpy; print(numpy.__version__)"` shows 2.3.4
- [ ] `python3 -c "import fastapi; print(fastapi.__version__)"` shows 0.121.0

### Test Verification
- [ ] Whisper service: `pytest tests/ -v` passes
- [ ] Orchestration service: `pytest tests/ -v` passes
- [ ] Translation service: `pytest tests/ -v` passes
- [ ] Bot container: `pytest tests/ -v` passes
- [ ] Integration tests: `pytest tests/integration/ -v` passes

### Functionality Verification
- [ ] Audio processing works (NumPy 2.x compatible)
- [ ] API endpoints respond correctly
- [ ] WebSocket connections stable
- [ ] Redis operations work
- [ ] Database queries execute
- [ ] Coverage reports generate correctly

### Performance Verification
- [ ] No performance regression
- [ ] NumPy operations fast
- [ ] FastAPI request handling performant
- [ ] Redis operations fast
- [ ] Test suite runs in reasonable time

---

## Success Metrics

After completing installation and testing, verify:

1. **Zero Installation Errors:** All packages install cleanly
2. **100% Test Pass Rate:** All existing tests pass (may need updates for NumPy 2.x)
3. **Coverage Accuracy:** pytest-cov 7.0.0 reports accurate coverage
4. **No Performance Regression:** Services perform as well or better
5. **No Breaking Changes:** All API endpoints work as before

---

## Next Steps After Installation

1. **Update CI/CD Pipelines**
   - Update GitHub Actions with new pytest 8.4.2
   - Update coverage thresholds for pytest-cov 7.0.0
   - Test parallel execution with pytest-xdist 3.8.0

2. **Monitor Production**
   - Watch for NumPy 2.x compatibility warnings
   - Monitor Redis connection stability
   - Track FastAPI request handling performance
   - Verify Pydantic validation accuracy

3. **Document Changes**
   - Update service READMEs with new versions
   - Document NumPy 2.x migration notes
   - Create dependency update policy
   - Establish security advisory monitoring

---

## Rollback Plan

If critical issues found during testing:

```bash
# Option 1: Rollback specific service
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/{service}
git checkout HEAD -- requirements*.txt pyproject.toml
pip install -r requirements.txt --force-reinstall

# Option 2: Rollback all services
cd /Users/thomaspatane/Documents/GitHub/livetranslate
git checkout HEAD -- "modules/*/requirements*.txt" "modules/*/pyproject.toml"

# Option 3: Rollback specific package (e.g., NumPy)
pip install "numpy<2.0" --force-reinstall
# Then manually revert numpy version in requirements files
```

---

## Support & Documentation

- **Full Report:** `/Users/thomaspatane/Documents/GitHub/livetranslate/DEPENDENCY_STANDARDIZATION_REPORT.md`
- **Change Summary:** `/Users/thomaspatane/Documents/GitHub/livetranslate/DEPENDENCY_CHANGES_SUMMARY.md`
- **Validation Script:** `/Users/thomaspatane/Documents/GitHub/livetranslate/scripts/validate_dependencies.sh`

---

**Status:** ✅ Ready for Testing
**Estimated Testing Time:** 1-2 days full validation
**Risk Level:** MEDIUM (NumPy 2.x requires thorough testing)
**Recommendation:** Test in staging environment before production deployment
