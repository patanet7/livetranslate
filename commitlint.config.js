/**
 * Commitlint Configuration for LiveTranslate
 *
 * Enforces conventional commit format:
 * <type>(<scope>): <description>
 *
 * Examples:
 * - feat(whisper): add NPU acceleration support
 * - fix(frontend): resolve audio recording issue
 * - docs(orchestration): update API documentation
 * - chore(deps): update dependencies
 *
 * @see https://commitlint.js.org/
 * @see https://www.conventionalcommits.org/
 */

module.exports = {
  extends: ["@commitlint/config-conventional"],

  rules: {
    // Type must be one of the following
    "type-enum": [
      2,
      "always",
      [
        "feat", // New feature
        "fix", // Bug fix
        "docs", // Documentation only changes
        "style", // Code style changes (formatting, semicolons, etc.)
        "refactor", // Code change that neither fixes a bug nor adds a feature
        "perf", // Performance improvement
        "test", // Adding or updating tests
        "build", // Build system or external dependencies
        "ci", // CI/CD configuration changes
        "chore", // Other changes that don't modify src or test files
        "revert", // Reverts a previous commit
        "security", // Security improvements
        "deps", // Dependency updates
        "wip", // Work in progress (should not be merged)
      ],
    ],

    // Type must be lowercase
    "type-case": [2, "always", "lower-case"],

    // Type cannot be empty
    "type-empty": [2, "never"],

    // Scope must be one of the following (optional)
    "scope-enum": [
      1, // Warning level - allows flexibility
      "always",
      [
        // Services
        "whisper",
        "whisper-service",
        "translation",
        "translation-service",
        "orchestration",
        "orchestration-service",
        "frontend",
        "frontend-service",
        "bot",
        "meeting-bot",

        // Infrastructure
        "ci",
        "cd",
        "docker",
        "k8s",
        "kubernetes",
        "helm",
        "terraform",
        "infra",
        "infrastructure",

        // Cross-cutting concerns
        "deps",
        "dependencies",
        "security",
        "auth",
        "api",
        "websocket",
        "ws",
        "audio",
        "database",
        "db",
        "redis",
        "config",
        "logging",
        "monitoring",
        "metrics",

        // Hardware acceleration
        "gpu",
        "npu",
        "cuda",

        // Development
        "dev",
        "test",
        "tests",
        "e2e",
        "integration",
        "unit",
        "lint",
        "format",
        "types",
        "typing",

        // Documentation
        "docs",
        "readme",
        "changelog",

        // Release
        "release",
        "version",
        "workflow",

        // Package managers
        "npm",
        "pip",
        "pdm",
        "pnpm",

        // Other
        "all",
        "core",
        "utils",
        "common",
        "shared",
        "secrets",
      ],
    ],

    // Scope must be lowercase
    "scope-case": [2, "always", "lower-case"],

    // Subject (description) rules
    "subject-case": [
      2,
      "never",
      ["sentence-case", "start-case", "pascal-case", "upper-case"],
    ],
    "subject-empty": [2, "never"],
    "subject-full-stop": [2, "never", "."],
    "subject-max-length": [2, "always", 72],
    "subject-min-length": [2, "always", 10],

    // Header (type + scope + subject) max length
    "header-max-length": [2, "always", 100],

    // Body rules
    "body-leading-blank": [2, "always"],
    "body-max-line-length": [1, "always", 100],

    // Footer rules
    "footer-leading-blank": [2, "always"],
    "footer-max-line-length": [1, "always", 100],

    // References (issue numbers)
    "references-empty": [1, "never"],
  },

  // Parser presets (for breaking changes, etc.)
  parserPreset: {
    parserOpts: {
      headerPattern: /^(\w*)(?:\(([^)]*)\))?!?: (.*)$/,
      headerCorrespondence: ["type", "scope", "subject"],
      noteKeywords: ["BREAKING CHANGE", "BREAKING-CHANGE"],
      revertPattern:
        /^(?:Revert|revert:)\s"?([\s\S]+?)"?\s*This reverts commit (\w*)\./i,
      revertCorrespondence: ["header", "hash"],
    },
  },

  // Custom prompt configuration for interactive commits
  prompt: {
    questions: {
      type: {
        description: "Select the type of change you're committing",
        enum: {
          feat: {
            description: "A new feature",
            title: "Features",
            emoji: "sparkles",
          },
          fix: {
            description: "A bug fix",
            title: "Bug Fixes",
            emoji: "bug",
          },
          docs: {
            description: "Documentation only changes",
            title: "Documentation",
            emoji: "memo",
          },
          style: {
            description:
              "Changes that do not affect the meaning of the code (formatting, etc.)",
            title: "Styles",
            emoji: "lipstick",
          },
          refactor: {
            description:
              "A code change that neither fixes a bug nor adds a feature",
            title: "Code Refactoring",
            emoji: "recycle",
          },
          perf: {
            description: "A code change that improves performance",
            title: "Performance Improvements",
            emoji: "zap",
          },
          test: {
            description: "Adding missing tests or correcting existing tests",
            title: "Tests",
            emoji: "white_check_mark",
          },
          build: {
            description:
              "Changes that affect the build system or external dependencies",
            title: "Builds",
            emoji: "package",
          },
          ci: {
            description: "Changes to CI/CD configuration files and scripts",
            title: "Continuous Integration",
            emoji: "construction_worker",
          },
          chore: {
            description: "Other changes that don't modify src or test files",
            title: "Chores",
            emoji: "wrench",
          },
          revert: {
            description: "Reverts a previous commit",
            title: "Reverts",
            emoji: "rewind",
          },
        },
      },
      scope: {
        description:
          "What is the scope of this change (e.g., whisper, frontend)",
      },
      subject: {
        description:
          "Write a short, imperative tense description of the change",
      },
      body: {
        description: "Provide a longer description of the change",
      },
      isBreaking: {
        description: "Are there any breaking changes?",
      },
      breakingBody: {
        description: "Describe the breaking changes",
      },
      isIssueAffected: {
        description: "Does this change affect any open issues?",
      },
      issues: {
        description: 'Add issue references (e.g., "fix #123", "re #456")',
      },
    },
  },

  // Help URL for commit message format
  helpUrl:
    "https://github.com/livetranslate/livetranslate/blob/main/CONTRIBUTING.md#commit-messages",

  // Default values
  defaultIgnores: true,

  // Ignore patterns for generated commits
  ignores: [
    (commit) => commit.includes("[skip ci]"),
    (commit) => commit.includes("[ci skip]"),
    (commit) => commit.startsWith("Merge"),
    (commit) => commit.startsWith("Revert"),
    (commit) => commit.includes("dependabot"),
    (commit) => commit.includes("renovate"),
  ],
};
