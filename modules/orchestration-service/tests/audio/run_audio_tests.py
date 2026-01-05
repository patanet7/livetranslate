#!/usr/bin/env python3
"""
Comprehensive Audio Testing Suite Runner

Orchestrates the complete audio testing suite including unit tests, integration tests,
and performance tests with I/O validation and component interaction verification.
"""

import asyncio
import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class TestResults:
    """Test results container."""

    suite_name: str
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    coverage_percent: float
    details: List[str]
    timestamp: str


@dataclass
class TestSuiteConfig:
    """Test suite configuration."""

    name: str
    test_path: str
    markers: List[str]
    timeout: int
    required_coverage: float
    performance_thresholds: Dict[str, float]


class AudioTestRunner:
    """Comprehensive audio test runner."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.results: List[TestResults] = []
        self.start_time = None

        # Test suite configurations
        self.test_suites = {
            "unit": TestSuiteConfig(
                name="Unit Tests",
                test_path="unit/",
                markers=["unit"],
                timeout=300,  # 5 minutes
                required_coverage=85.0,
                performance_thresholds={"max_test_duration": 0.1},
            ),
            "integration": TestSuiteConfig(
                name="Integration Tests",
                test_path="integration/",
                markers=["integration"],
                timeout=600,  # 10 minutes
                required_coverage=75.0,
                performance_thresholds={"max_test_duration": 5.0},
            ),
            "performance": TestSuiteConfig(
                name="Performance Tests",
                test_path="performance/",
                markers=["performance"],
                timeout=1200,  # 20 minutes
                required_coverage=60.0,
                performance_thresholds={
                    "throughput_ratio": 5.0,
                    "max_latency_ms": 100,
                    "max_memory_mb": 200,
                },
            ),
            "e2e": TestSuiteConfig(
                name="End-to-End Tests",
                test_path="e2e/",
                markers=["e2e"],
                timeout=900,  # 15 minutes
                required_coverage=50.0,
                performance_thresholds={"max_test_duration": 10.0},
            ),
        }

    def run_test_suite(
        self, suite_key: str, verbose: bool = False, capture_output: bool = True
    ) -> TestResults:
        """Run a specific test suite."""
        config = self.test_suites[suite_key]
        test_path = self.base_path / config.test_path

        print(f"\n🧪 Running {config.name}...")
        print(f"   Path: {test_path}")
        print(f"   Timeout: {config.timeout}s")

        if not test_path.exists():
            print(f"   ⚠️  Test path does not exist: {test_path}")
            return TestResults(
                suite_name=config.name,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=0.0,
                coverage_percent=0.0,
                details=[f"Test path does not exist: {test_path}"],
                timestamp=datetime.now().isoformat(),
            )

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "--tb=short",
            "--durations=10",
            "--cov=src.audio",
            "--cov-report=term-missing",
            "--cov-report=json",
            "--json-report",
            "--json-report-file=test_results.json",
        ]

        # Add markers if specified
        if config.markers:
            for marker in config.markers:
                cmd.extend(["-m", marker])

        if verbose:
            cmd.append("-v")

        if not capture_output:
            cmd.append("-s")

        # Run tests
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_path,
                capture_output=capture_output,
                text=True,
                timeout=config.timeout,
            )

            duration = time.time() - start_time

            # Parse results
            test_results = self._parse_pytest_results(
                config.name, result, duration, verbose
            )

            # Check coverage
            coverage_data = self._get_coverage_data()
            if coverage_data:
                test_results.coverage_percent = coverage_data.get("totals", {}).get(
                    "percent_covered", 0.0
                )

            # Validate against requirements
            self._validate_test_results(test_results, config)

            return test_results

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResults(
                suite_name=config.name,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=duration,
                coverage_percent=0.0,
                details=[f"Test suite timed out after {config.timeout}s"],
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResults(
                suite_name=config.name,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration=duration,
                coverage_percent=0.0,
                details=[f"Test execution error: {str(e)}"],
                timestamp=datetime.now().isoformat(),
            )

    def _parse_pytest_results(
        self,
        suite_name: str,
        result: subprocess.CompletedProcess,
        duration: float,
        verbose: bool,
    ) -> TestResults:
        """Parse pytest results from JSON report."""
        details = []

        # Try to load JSON report
        json_file = self.base_path / "test_results.json"
        passed = failed = errors = skipped = 0

        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                summary = data.get("summary", {})
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                errors = summary.get("error", 0)
                skipped = summary.get("skipped", 0)

                # Add test details
                if verbose and "tests" in data:
                    for test in data["tests"]:
                        if test.get("outcome") in ["failed", "error"]:
                            details.append(f"FAILED: {test.get('nodeid', 'unknown')}")
                            if "call" in test and "longrepr" in test["call"]:
                                details.append(f"  Error: {test['call']['longrepr']}")

                # Clean up JSON file
                json_file.unlink()

            except Exception as e:
                details.append(f"Failed to parse JSON report: {e}")

        # Fallback to parsing stdout/stderr
        if passed + failed + errors + skipped == 0:
            output = result.stdout + result.stderr
            details.append(f"Return code: {result.returncode}")

            if result.returncode == 0:
                passed = 1  # Assume success if no detailed info
            else:
                failed = 1

            if output and verbose:
                details.extend(output.split("\n")[-20:])  # Last 20 lines

        return TestResults(
            suite_name=suite_name,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration=duration,
            coverage_percent=0.0,  # Will be updated by caller
            details=details,
            timestamp=datetime.now().isoformat(),
        )

    def _get_coverage_data(self) -> Optional[Dict]:
        """Get coverage data from JSON report."""
        coverage_file = self.base_path / "coverage.json"

        if coverage_file.exists():
            try:
                with open(coverage_file, "r") as f:
                    data = json.load(f)
                coverage_file.unlink()  # Clean up
                return data
            except Exception:
                pass

        return None

    def _validate_test_results(self, results: TestResults, config: TestSuiteConfig):
        """Validate test results against configuration requirements."""
        # Check coverage requirement
        if results.coverage_percent < config.required_coverage:
            results.details.append(
                f"⚠️  Coverage {results.coverage_percent:.1f}% below required {config.required_coverage}%"
            )

        # Check performance thresholds
        for threshold_name, threshold_value in config.performance_thresholds.items():
            if (
                threshold_name == "max_test_duration"
                and results.duration > threshold_value
            ):
                results.details.append(
                    f"⚠️  Test duration {results.duration:.1f}s exceeds threshold {threshold_value}s"
                )

    def run_comprehensive_suite(
        self,
        suites: List[str] = None,
        verbose: bool = False,
        fail_fast: bool = False,
        capture_output: bool = True,
    ) -> Dict[str, TestResults]:
        """Run comprehensive test suite."""
        if suites is None:
            suites = ["unit", "integration", "performance"]

        self.start_time = time.time()
        results = {}

        print("🚀 Starting Comprehensive Audio Testing Suite")
        print(f"   Suites: {', '.join(suites)}")
        print(f"   Base path: {self.base_path}")
        print(f"   Timestamp: {datetime.now().isoformat()}")

        # Environment validation
        self._validate_environment()

        for suite_key in suites:
            if suite_key not in self.test_suites:
                print(f"❌ Unknown test suite: {suite_key}")
                continue

            suite_results = self.run_test_suite(suite_key, verbose, capture_output)
            results[suite_key] = suite_results
            self.results.append(suite_results)

            # Print suite summary
            self._print_suite_summary(suite_results)

            # Fail fast if requested and tests failed
            if fail_fast and (suite_results.failed > 0 or suite_results.errors > 0):
                print(f"💥 Failing fast due to failures in {suite_results.suite_name}")
                break

        # Print overall summary
        self._print_overall_summary(results)

        return results

    def _validate_environment(self):
        """Validate test environment."""
        print("\n🔍 Validating test environment...")

        # Check Python packages
        required_packages = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-json-report",
            "numpy",
            "scipy",
            "soundfile",
            "httpx",
            "psutil",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"   ⚠️  Missing packages: {', '.join(missing_packages)}")
            print("   Install with: pip install " + " ".join(missing_packages))
        else:
            print("   ✅ All required packages available")

        # Check test data directories
        test_dirs = ["unit", "integration", "performance"]
        for test_dir in test_dirs:
            dir_path = self.base_path / test_dir
            if dir_path.exists():
                test_files = list(dir_path.glob("test_*.py"))
                print(f"   ✅ {test_dir}: {len(test_files)} test files")
            else:
                print(f"   ⚠️  {test_dir}: directory missing")

    def _print_suite_summary(self, results: TestResults):
        """Print summary for a test suite."""
        total_tests = results.passed + results.failed + results.errors + results.skipped

        # Determine status emoji
        if results.errors > 0:
            status = "💥"
        elif results.failed > 0:
            status = "❌"
        elif total_tests == 0:
            status = "⚪"
        else:
            status = "✅"

        print(f"\n{status} {results.suite_name}")
        print(
            f"   Tests: {results.passed}✅ {results.failed}❌ {results.errors}💥 {results.skipped}⏭️"
        )
        print(f"   Duration: {results.duration:.1f}s")
        print(f"   Coverage: {results.coverage_percent:.1f}%")

        # Print details if there are issues
        if results.details and (results.failed > 0 or results.errors > 0):
            print("   Details:")
            for detail in results.details[-5:]:  # Last 5 details
                print(f"     {detail}")

    def _print_overall_summary(self, results: Dict[str, TestResults]):
        """Print overall test summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        # Calculate totals
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_skipped = sum(r.skipped for r in results.values())
        total_tests = total_passed + total_failed + total_errors + total_skipped

        # Calculate average coverage
        avg_coverage = (
            sum(r.coverage_percent for r in results.values()) / len(results)
            if results
            else 0
        )

        # Determine overall status
        if total_errors > 0:
            overall_status = "💥 FAILED"
        elif total_failed > 0:
            overall_status = "❌ FAILED"
        elif total_tests == 0:
            overall_status = "⚪ NO TESTS"
        else:
            overall_status = "✅ PASSED"

        print(f"\n" + "=" * 60)
        print(f"🏁 COMPREHENSIVE AUDIO TESTING SUMMARY")
        print(f"   Status: {overall_status}")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed} ✅")
        print(f"   Failed: {total_failed} ❌")
        print(f"   Errors: {total_errors} 💥")
        print(f"   Skipped: {total_skipped} ⏭️")
        print(f"   Average Coverage: {avg_coverage:.1f}%")
        print(f"   Total Duration: {total_duration:.1f}s")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)

        # Success rate
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"   Success Rate: {success_rate:.1f}%")

            if success_rate >= 95:
                print("   🎉 Excellent test results!")
            elif success_rate >= 85:
                print("   👍 Good test results!")
            elif success_rate >= 70:
                print("   ⚠️  Test results need improvement")
            else:
                print("   🚨 Poor test results - investigate failures")

    def generate_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate comprehensive test report."""
        if output_path is None:
            output_path = (
                self.base_path
                / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "test_runner_version": "1.0.0",
            "base_path": str(self.base_path),
            "total_duration": time.time() - self.start_time if self.start_time else 0,
            "suite_results": [asdict(result) for result in self.results],
            "summary": {
                "total_suites": len(self.results),
                "total_tests": sum(
                    r.passed + r.failed + r.errors + r.skipped for r in self.results
                ),
                "total_passed": sum(r.passed for r in self.results),
                "total_failed": sum(r.failed for r in self.results),
                "total_errors": sum(r.errors for r in self.results),
                "total_skipped": sum(r.skipped for r in self.results),
                "average_coverage": sum(r.coverage_percent for r in self.results)
                / len(self.results)
                if self.results
                else 0,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Test report generated: {output_path}")
        return output_path


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive Audio Testing Suite")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["unit", "integration", "performance", "e2e"],
        default=["unit", "integration", "performance"],
        help="Test suites to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fail-fast", "-x", action="store_true", help="Stop on first failure"
    )
    parser.add_argument(
        "--no-capture", "-s", action="store_true", help="Don't capture output"
    )
    parser.add_argument("--report", help="Generate report to specified file")
    parser.add_argument("--base-path", help="Base path for tests")

    args = parser.parse_args()

    # Setup base path
    base_path = Path(args.base_path) if args.base_path else Path(__file__).parent

    # Create test runner
    runner = AudioTestRunner(base_path)

    # Run tests
    try:
        results = runner.run_comprehensive_suite(
            suites=args.suites,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
            capture_output=not args.no_capture,
        )

        # Generate report if requested
        if args.report:
            runner.generate_report(Path(args.report))

        # Exit with appropriate code
        total_failures = sum(r.failed + r.errors for r in results.values())
        sys.exit(0 if total_failures == 0 else 1)

    except KeyboardInterrupt:
        print("\n⚡ Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
