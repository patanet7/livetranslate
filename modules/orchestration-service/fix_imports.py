#!/usr/bin/env python3
"""
Fix relative imports to absolute imports with src. prefix
This script updates all Python files in the orchestration-service to use absolute imports.
"""

import re
import sys
from pathlib import Path

# Modules that need src. prefix
MODULES_TO_FIX = [
    "models",
    "dependencies",
    "managers",
    "clients",
    "audio",
    "database",
    "pipeline",
    "utils",
    "infrastructure",
    "config",
    "bot",
]


def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            fixed_line = line

            # Pattern 1: from models import ... or from models. import ...
            for module in MODULES_TO_FIX:
                # Don't fix if already has src. prefix
                if f"from src.{module}" in line:
                    fixed_line = line
                    break

                # Fix: from models import X
                pattern1 = f"^(\\s*)from {module} import"
                if re.match(pattern1, line):
                    fixed_line = re.sub(f"from {module} import", f"from src.{module} import", line)
                    break

                # Fix: from models.submodule import X
                pattern2 = f"^(\\s*)from {module}\\."
                if re.match(pattern2, line):
                    fixed_line = re.sub(f"from {module}\\.", f"from src.{module}.", line)
                    break

            fixed_lines.append(fixed_line)

        fixed_content = "\n".join(fixed_lines)

        # Only write if something changed
        if fixed_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Fix all Python files in src directory."""
    src_dir = Path(__file__).parent / "src"

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    python_files = list(src_dir.rglob("*.py"))
    fixed_count = 0

    print(f"Found {len(python_files)} Python files")

    for py_file in python_files:
        if "__pycache__" in str(py_file):
            continue

        if fix_imports_in_file(py_file):
            fixed_count += 1
            print(f"✓ Fixed: {py_file.relative_to(src_dir.parent)}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
