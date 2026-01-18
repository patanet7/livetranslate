## Description

<!-- Provide a clear and concise description of what this PR does. -->

## Related Issues

<!-- Link related issues using keywords like "Fixes #123" or "Relates to #456" -->

- Fixes #
- Relates to #

## Type of Change

<!-- Mark the appropriate option with an [x] -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)
- [ ] Documentation update
- [ ] CI/CD changes
- [ ] Dependency update
- [ ] Security fix

## Service(s) Affected

<!-- Mark all services affected by this change -->

- [ ] Whisper Service (`modules/whisper-service/`)
- [ ] Translation Service (`modules/translation-service/`)
- [ ] Orchestration Service (`modules/orchestration-service/`)
- [ ] Frontend Service (`modules/frontend-service/`)
- [ ] Infrastructure/CI/CD (`.github/`, `docker/`, etc.)
- [ ] Documentation (`docs/`, `*.md`)

## Testing

### Test Coverage

<!-- Describe the tests you ran to verify your changes -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed

### Test Commands Run

```bash
# List the test commands you ran
pdm run pytest tests/
pnpm test
```

### Test Results

<!-- Include relevant test output or screenshots -->

## Checklist

### Code Quality

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings or errors

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have verified the changes work as expected in a local environment

### Security

- [ ] I have not introduced any security vulnerabilities
- [ ] I have not committed any secrets, API keys, or sensitive data
- [ ] I have reviewed the security implications of my changes

### Dependencies

- [ ] I have updated `pyproject.toml` / `package.json` if dependencies changed
- [ ] I have verified new dependencies are compatible with our license requirements
- [ ] I have checked new dependencies for known vulnerabilities

## Screenshots/Recordings

<!-- If applicable, add screenshots or recordings to help explain your changes -->

## Performance Impact

<!-- Describe any performance implications of your changes -->

- [ ] No significant performance impact
- [ ] Performance improvement (describe below)
- [ ] Potential performance degradation (justified below)

## Deployment Notes

<!-- Any special considerations for deploying this change? -->

- [ ] No special deployment steps required
- [ ] Database migration required
- [ ] Configuration changes required
- [ ] Feature flag required
- [ ] Other (describe below)

## Additional Notes

<!-- Any other information that reviewers should know? -->
