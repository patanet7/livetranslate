# ADR 0002: PDM for Dependency Management

## Status

Accepted

## Context

The project was using Poetry for Python dependency management. As the project grew, we identified several issues:
- Poetry lock file resolution can be slow
- Poetry doesn't fully support PEP 582 (__pypackages__)
- Inconsistent lock file formats between Poetry versions
- Need for better monorepo support

## Decision

We will migrate from Poetry to PDM (Python Dependency Manager) for all Python services.

## Rationale

PDM offers several advantages:

1. **PEP Compliance**: Full support for PEP 517, 582, 621
2. **Speed**: Faster dependency resolution
3. **Flexibility**: Better support for monorepos and workspaces
4. **Standards**: Uses standard pyproject.toml format
5. **Modern**: Active development and community support

## Implementation

1. Update all pyproject.toml files to use PDM build system
2. Convert Poetry dependency groups to PDM format
3. Update CI/CD pipelines to use `pdm install`
4. Update Docker images to install PDM
5. Update development scripts

## Consequences

### Positive
- Faster dependency resolution
- Better PEP compliance
- Cleaner pyproject.toml files
- Consistent with Python standards

### Negative
- Learning curve for developers familiar with Poetry
- Need to update existing CI/CD pipelines
- Documentation updates required

## Related Decisions
- ADR 0001: Microservices Architecture
