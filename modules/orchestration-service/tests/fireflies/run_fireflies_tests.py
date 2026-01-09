#!/usr/bin/env python3
"""
Fireflies Test Runner

Run all Fireflies integration tests with comprehensive output.
Results are saved to tests/output/ with timestamp.

Usage:
    python tests/fireflies/run_fireflies_tests.py
    python tests/fireflies/run_fireflies_tests.py --unit
    python tests/fireflies/run_fireflies_tests.py --integration
    python tests/fireflies/run_fireflies_tests.py -v
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess
import argparse

# Get paths
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATION_ROOT = SCRIPT_DIR.parent.parent
TESTS_DIR = ORCHESTRATION_ROOT / "tests"
OUTPUT_DIR = TESTS_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_tests(test_type: str = "all", verbose: bool = False, output_file: str = None):
    """
    Run Fireflies tests.

    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration')
        verbose: Enable verbose output
        output_file: Path to output file
    """
    # Build pytest command
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        str(SCRIPT_DIR),
        "--tb=short",
        "-v" if verbose else "-q",
    ]

    # Add test type filter
    if test_type == "unit":
        pytest_args.extend(["-k", "unit"])
    elif test_type == "integration":
        pytest_args.extend(["-k", "integration"])

    # Add markers
    pytest_args.extend(["-m", "not slow"])

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"{timestamp}_test_fireflies_results.log"

    print("=" * 70)
    print("Fireflies Test Runner")
    print("=" * 70)
    print(f"Test type: {test_type}")
    print(f"Output file: {output_file}")
    print(f"Verbose: {verbose}")
    print("=" * 70)
    print()

    # Run tests and capture output
    with open(output_file, "w") as f:
        f.write("Fireflies Test Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Test type: {test_type}\n")
        f.write("=" * 70 + "\n\n")

        # Run pytest
        result = subprocess.run(
            pytest_args,
            cwd=ORCHESTRATION_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Write output
        f.write(result.stdout)
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Exit code: {result.returncode}\n")

        # Also print to console
        print(result.stdout)

    print()
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Exit code: {result.returncode}")
    print("=" * 70)

    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Fireflies tests")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path",
    )

    args = parser.parse_args()

    # Determine test type
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    else:
        test_type = "all"

    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        verbose=args.verbose,
        output_file=args.output,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
