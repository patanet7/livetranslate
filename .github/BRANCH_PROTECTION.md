# Branch Protection Rules

This document describes the recommended branch protection rules for the LiveTranslate repository.

## Main Branch Protection

The `main` branch should be protected with the following rules to ensure code quality and prevent accidental changes.

### Required Settings

#### 1. Require Pull Request Reviews

- **Required approving reviews:** 1 (minimum)
- **Dismiss stale pull request approvals when new commits are pushed:** Enabled
- **Require review from Code Owners:** Enabled
- **Restrict who can dismiss pull request reviews:** Enabled (maintainers only)

#### 2. Require Status Checks

The following status checks must pass before merging:

**Python Services:**
- `lint-python / whisper-service`
- `lint-python / translation-service`
- `lint-python / orchestration-service`
- `type-check-python / whisper-service`
- `type-check-python / translation-service`
- `type-check-python / orchestration-service`
- `test-python / whisper-service`
- `test-python / translation-service`
- `test-python / orchestration-service`

**Frontend Service:**
- `lint-frontend`
- `type-check-frontend`
- `test-frontend`
- `build-frontend`

**Security:**
- `bandit-scan`
- `secret-scan`
- `codeql-analysis / python`
- `codeql-analysis / javascript`

**Additional Settings:**
- **Require branches to be up to date before merging:** Enabled
- **Require status checks to pass before merging:** Enabled

#### 3. Require Conversation Resolution

- **Require conversation resolution before merging:** Enabled

#### 4. Require Linear History

- **Require linear history:** Enabled
- This enforces squash or rebase merges, preventing merge commits

#### 5. Restrict Force Pushes

- **Do not allow force pushes:** Enabled
- Prevents rewriting history on the main branch

#### 6. Restrict Deletions

- **Do not allow deletions:** Enabled
- Prevents accidental deletion of the main branch

### Optional Settings

#### Require Signed Commits

- **Require signed commits:** Recommended for production environments
- Ensures all commits are cryptographically signed

#### Include Administrators

- **Include administrators:** Recommended
- Ensures that even repository administrators must follow the protection rules

## Setting Up Branch Protection

### Via GitHub UI

1. Navigate to **Settings** > **Branches**
2. Click **Add branch protection rule**
3. Enter `main` as the branch name pattern
4. Configure the settings as described above
5. Click **Create** or **Save changes**

### Via GitHub CLI

```bash
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci-status"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":true,"required_approving_review_count":1}' \
  --field restrictions=null \
  --field required_linear_history=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

### Via Terraform (GitHub Provider)

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.livetranslate.node_id
  pattern       = "main"

  required_status_checks {
    strict   = true
    contexts = [
      "lint-python",
      "type-check-python",
      "test-python",
      "lint-frontend",
      "type-check-frontend",
      "test-frontend",
      "build-frontend",
      "ci-status"
    ]
  }

  required_pull_request_reviews {
    dismiss_stale_reviews           = true
    require_code_owner_reviews      = true
    required_approving_review_count = 1
  }

  require_conversation_resolution = true
  required_linear_history         = true
  allows_force_pushes            = false
  allows_deletions               = false
}
```

## Release Branch Protection

For release branches (e.g., `release/*`), consider:

- Same protection rules as main
- Additional approval requirement (2 reviewers)
- Restrict who can push to release branches

## Feature Branch Guidelines

While feature branches are not protected, follow these guidelines:

1. **Naming Convention:** `feature/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`
2. **Keep branches short-lived:** Merge within 1-2 weeks
3. **Rebase regularly:** Keep up to date with main
4. **Delete after merge:** Clean up merged branches

## Enforcement

Branch protection rules are enforced at the GitHub level and cannot be bypassed without administrator intervention. This ensures:

- All code changes go through code review
- All tests pass before merging
- No unauthorized changes to protected branches
- Complete audit trail of all changes

## Exceptions

In rare circumstances (e.g., critical hotfixes), administrators may temporarily disable protection. This should:

1. Be documented in the PR/commit message
2. Be re-enabled immediately after the change
3. Be reviewed in the next team retrospective
