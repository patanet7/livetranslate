# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- DX Optimization: Comprehensive developer experience improvements
- PDM migration: Switched from Poetry to PDM for dependency management
- Pre-commit hooks: Enhanced security and quality checks
- CI/CD pipeline: GitHub Actions workflows for testing and deployment
- Cross-platform scripts: Bash scripts for macOS/Linux development
- IDE configuration: VSCode settings, launch configurations, and extensions

### Changed
- Standardized Python version to >=3.12,<3.15 across all services
- Standardized line length to 100 characters
- Replaced flake8 with Ruff for faster linting
- Updated Docker images with non-root users and health checks

### Fixed
- Audio pipeline 422 validation errors
- Model name standardization ("whisper-base" naming)

### Security
- Added non-root users to all Docker containers
- Removed chmod 777 security issues
- Added secret scanning with detect-secrets and gitleaks

## [1.0.0] - 2026-01-17

### Added
- Initial release of LiveTranslate
- Whisper Service with NPU optimization
- Translation Service with GPU optimization
- Orchestration Service with bot management
- Frontend Service with React 18 + TypeScript
- Google Meet bot integration
- Virtual webcam with speaker attribution
- Real-time WebSocket infrastructure

[Unreleased]: https://github.com/your-org/livetranslate/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/livetranslate/releases/tag/v1.0.0
