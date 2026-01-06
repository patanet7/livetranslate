#!/bin/bash
# Dependency Validation Script
# Validates that all standardized dependencies can be installed successfully

set -e  # Exit on error

echo "=========================================="
echo "LiveTranslate Dependency Validation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
WARNINGS=0

# Function to test package installation
test_package() {
    local package=$1
    local version=$2
    echo -n "Testing ${package}>=${version}... "

    if python -m pip install --dry-run "${package}>=${version}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

# Function to test package compatibility
test_compatibility() {
    local pkg1=$1
    local ver1=$2
    local pkg2=$3
    local ver2=$4
    echo -n "Testing ${pkg1}>=${ver1} + ${pkg2}>=${ver2}... "

    if python -m pip install --dry-run "${pkg1}>=${ver1}" "${pkg2}>=${ver2}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ COMPATIBLE${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠ WARNING${NC}"
        ((WARNINGS++))
        return 1
    fi
}

echo "1. Testing Core Testing Packages"
echo "=================================="
test_package "pytest" "8.4.2"
test_package "pytest-asyncio" "1.2.0"
test_package "pytest-cov" "7.0.0"
test_package "pytest-mock" "3.15.1"
test_package "pytest-xdist" "3.8.0"
test_package "hypothesis" "6.145.1"
echo ""

echo "2. Testing FastAPI Ecosystem"
echo "============================="
test_package "fastapi" "0.121.0"
test_package "uvicorn" "0.38.0"
test_package "pydantic" "2.12.3"
test_package "pydantic-settings" "2.7.1"
echo ""

echo "3. Testing WebSocket & Networking"
echo "=================================="
test_package "websockets" "15.0.1"
test_package "httpx" "0.28.1"
test_package "requests" "2.31.0"
test_package "python-socketio" "5.14.2"
test_package "aiohttp" "3.8.5"
echo ""

echo "4. Testing Data Processing"
echo "==========================="
test_package "numpy" "2.3.4"
test_package "scipy" "1.16.2"
test_package "pandas" "2.0.0"
test_package "librosa" "0.11.0"
test_package "soundfile" "0.13.1"
echo ""

echo "5. Testing Redis & Caching"
echo "==========================="
test_package "redis" "6.4.0"
test_package "fakeredis" "2.32.0"
echo ""

echo "6. Testing Database Packages"
echo "============================="
test_package "sqlalchemy" "2.0.44"
test_package "alembic" "1.17.0"
test_package "asyncpg" "0.30.0"
echo ""

echo "7. Testing Critical Compatibility"
echo "=================================="
test_compatibility "numpy" "2.3.4" "scipy" "1.16.2"
test_compatibility "numpy" "2.3.4" "librosa" "0.11.0"
test_compatibility "fastapi" "0.121.0" "pydantic" "2.12.3"
test_compatibility "pytest" "8.4.2" "pytest-asyncio" "1.2.0"
test_compatibility "pytest" "8.4.2" "pytest-cov" "7.0.0"
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL VALIDATIONS PASSED${NC}"
    echo "Dependencies are ready for installation."
    exit 0
else
    echo -e "${RED}✗ SOME VALIDATIONS FAILED${NC}"
    echo "Please review failed packages before proceeding."
    exit 1
fi
