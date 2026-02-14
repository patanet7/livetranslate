# Contributing to LiveTranslate

Thank you for your interest in contributing to LiveTranslate! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- PDM (Python Dependency Manager)
- Docker and Docker Compose
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/livetranslate.git
   cd livetranslate
   ```

2. **Install PDM** (if not already installed)
   ```bash
   pip install pdm
   ```

3. **Install dependencies for each service**
   ```bash
   # Orchestration Service
   cd modules/orchestration-service
   pdm install

   # Whisper Service
   cd ../whisper-service
   pdm install

   # Translation Service
   cd ../translation-service
   pdm install

   # Frontend Service
   cd ../frontend-service
   npm install
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Copy environment templates**
   ```bash
   cp modules/whisper-service/.env.example modules/whisper-service/.env.local
   cp modules/translation-service/.env.example modules/translation-service/.env.local
   ```

## Making Changes

### Branch Naming Convention

- `feature/` - New features (e.g., `feature/add-language-detection`)
- `fix/` - Bug fixes (e.g., `fix/audio-sync-issue`)
- `docs/` - Documentation changes (e.g., `docs/update-api-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-pipeline`)
- `test/` - Test additions/changes (e.g., `test/add-integration-tests`)
- `chore/` - Maintenance tasks (e.g., `chore/update-dependencies`)

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Scopes:**
- `whisper`: Whisper service
- `translation`: Translation service
- `orchestration`: Orchestration service
- `frontend`: Frontend service
- `docker`: Docker/containerization
- `deps`: Dependencies

**Examples:**
```
feat(whisper): add NPU acceleration support
fix(translation): resolve memory leak in batch processing
docs(orchestration): update API documentation
```

## Testing Guidelines

### IMPORTANT: Behavioral Tests Only

All tests in this repository must be **behavioral/integration tests** that test real system behavior:

1. **NO MOCKING** - Do not mock services, databases, or external dependencies
2. **Real Services** - Tests should use real service instances
3. **Real Data Flow** - Test actual data flowing through the system
4. **Test Outputs** - All test results must go to `tests/output/` with format: `TIMESTAMP_test_XXX_results.log`

### Running Tests

```bash
# Run all tests for a service
cd modules/orchestration-service
pdm run pytest

# Run specific test markers
pdm run pytest -m "integration"
pdm run pytest -m "behavioral"

# Run with coverage
pdm run pytest --cov=src --cov-report=html
```

### Test Output Location

- Backend tests: `modules/<service>/tests/output/`
- Frontend tests: `modules/frontend-service/tests/output/`

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the style guidelines
3. **Write/update tests** - behavioral tests only!
4. **Run the test suite** locally
5. **Submit a pull request** using the PR template
6. **Address review feedback**
7. **Squash and merge** when approved

### PR Checklist

- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Documentation updated (if applicable)
- [ ] No new warnings introduced
- [ ] PR description clearly explains changes

## Style Guidelines

### Python

- **Line length**: 100 characters
- **Formatter**: Ruff (replaces Black + isort)
- **Linter**: Ruff
- **Type checker**: mypy (strict mode)
- **Python version**: 3.12+

### TypeScript/JavaScript

- **Formatter**: Prettier
- **Linter**: ESLint
- **Style**: Consistent with existing codebase

### Configuration

All services use standardized configuration:
- `pyproject.toml` for Python projects (PDM format)
- `ruff.toml` at root for shared Ruff config
- `.editorconfig` for cross-editor consistency

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

Thank you for contributing!
