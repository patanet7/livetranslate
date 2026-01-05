#!/usr/bin/env python3
"""
Comprehensive Audio Flow Test Runner

This script runs the complete suite of audio processing tests including
integration, performance, error handling, and regression testing.
"""

import asyncio
import logging
import argparse
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import platform

import pytest
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_audio_tests.log"),
    ],
)
logger = logging.getLogger(__name__)


class TestSuiteRunner:
    """Comprehensive test suite runner with reporting and monitoring."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.performance_metrics = {}
        self.error_summary = {}
        self.system_info = {}

    def collect_system_info(self):
        """Collect system information for test reporting."""
        self.system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_gb": psutil.disk_usage(".").free / (1024**3),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("System Information:")
        for key, value in self.system_info.items():
            logger.info(f"  {key}: {value}")

    def run_test_category(
        self,
        category: str,
        test_files: List[str],
        markers: List[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run a specific category of tests."""
        logger.info(f"Running {category} tests...")

        # Build pytest command
        cmd = ["python", "-m", "pytest"]

        # Add test files
        for test_file in test_files:
            cmd.append(test_file)

        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Add verbose output
        if verbose:
            cmd.append("-v")

        # Add output format
        cmd.extend(
            [
                "--tb=short",
                "--durations=10",
                f"--junitxml=test_results_{category}.xml",
                f"--html=test_report_{category}.html",
                "--self-contained-html",
            ]
        )

        # Run tests
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()

        # Parse results
        test_result = {
            "category": category,
            "exit_code": result.returncode,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd),
        }

        # Extract test counts from output
        if result.stdout:
            lines = result.stdout.split("\n")
            for line in lines:
                if "passed" in line and "failed" in line:
                    test_result["summary_line"] = line.strip()
                    break

        return test_result

    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests."""
        test_files = [
            "tests/integration/test_complete_audio_flow.py",
            "tests/integration/test_audio_coordinator_integration.py",
            "tests/integration/test_chunk_manager_integration.py",
        ]

        return self.run_test_category(
            "integration", test_files, markers=["not slow"], verbose=verbose
        )

    def run_performance_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run performance tests."""
        test_files = [
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_performance_benchmarks",
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_memory_usage_monitoring",
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_concurrent_session_processing",
            "tests/performance/test_audio_performance.py",
        ]

        return self.run_test_category(
            "performance", test_files, markers=["performance"], verbose=verbose
        )

    def run_error_handling_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run error handling tests."""
        test_files = [
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_error_scenarios_comprehensive",
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_service_failure_scenarios",
        ]

        return self.run_test_category(
            "error_handling", test_files, markers=["error"], verbose=verbose
        )

    def run_format_compatibility_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run format compatibility tests."""
        test_files = [
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_format_compatibility_all_formats",
            "tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_audio_quality_validation",
        ]

        return self.run_test_category(
            "format_compatibility", test_files, markers=["format"], verbose=verbose
        )

    def run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests."""
        test_files = [
            "tests/unit/test_audio_models.py",
            "tests/unit/test_audio_processor.py",
            "tests/unit/test_speaker_correlator.py",
            "tests/unit/test_timing_coordinator.py",
        ]

        return self.run_test_category("unit", test_files, verbose=verbose)

    def run_comprehensive_suite(
        self, categories: List[str] = None, verbose: bool = True, quick: bool = False
    ) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        self.start_time = time.time()

        # Collect system information
        self.collect_system_info()

        # Default categories
        if categories is None:
            categories = [
                "unit",
                "integration",
                "format_compatibility",
                "error_handling",
            ]

            if not quick:
                categories.extend(["performance"])

        results = {}

        # Run each category
        for category in categories:
            try:
                if category == "unit":
                    result = self.run_unit_tests(verbose)
                elif category == "integration":
                    result = self.run_integration_tests(verbose)
                elif category == "performance":
                    result = self.run_performance_tests(verbose)
                elif category == "error_handling":
                    result = self.run_error_handling_tests(verbose)
                elif category == "format_compatibility":
                    result = self.run_format_compatibility_tests(verbose)
                else:
                    logger.warning(f"Unknown test category: {category}")
                    continue

                results[category] = result

                # Log results
                if result["exit_code"] == 0:
                    logger.info(
                        f"✅ {category} tests PASSED ({result['duration']:.1f}s)"
                    )
                else:
                    logger.error(
                        f"❌ {category} tests FAILED ({result['duration']:.1f}s)"
                    )

                if "summary_line" in result:
                    logger.info(f"   {result['summary_line']}")

            except Exception as e:
                logger.error(f"Error running {category} tests: {e}")
                results[category] = {
                    "category": category,
                    "error": str(e),
                    "exit_code": 1,
                    "duration": 0,
                }

        self.end_time = time.time()
        self.results = results

        return results

    def generate_report(self, output_file: str = "comprehensive_test_report.json"):
        """Generate comprehensive test report."""
        if not self.results:
            logger.warning("No test results to report")
            return

        # Calculate summary statistics
        total_duration = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )
        passed_categories = [
            cat for cat, result in self.results.items() if result.get("exit_code") == 0
        ]
        failed_categories = [
            cat for cat, result in self.results.items() if result.get("exit_code") != 0
        ]

        report = {
            "test_run_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat()
                if self.start_time
                else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time
                else None,
                "total_duration": total_duration,
                "categories_tested": list(self.results.keys()),
                "categories_passed": passed_categories,
                "categories_failed": failed_categories,
                "overall_success": len(failed_categories) == 0,
            },
            "system_info": self.system_info,
            "test_results": self.results,
            "performance_metrics": self.performance_metrics,
            "error_summary": self.error_summary,
        }

        # Write report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Test report saved to {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("COMPREHENSIVE AUDIO TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Duration: {total_duration:.1f}s")
        print(f"Categories Tested: {len(self.results)}")
        print(f"Categories Passed: {len(passed_categories)}")
        print(f"Categories Failed: {len(failed_categories)}")
        print(
            f"Overall Result: {'✅ PASS' if len(failed_categories) == 0 else '❌ FAIL'}"
        )

        print("\nCategory Results:")
        for category, result in self.results.items():
            status = "✅ PASS" if result.get("exit_code") == 0 else "❌ FAIL"
            duration = result.get("duration", 0)
            print(f"  {category:20} {status:8} ({duration:.1f}s)")

        if failed_categories:
            print("\nFailed Categories:")
            for category in failed_categories:
                result = self.results[category]
                print(f"  {category}:")
                if "error" in result:
                    print(f"    Error: {result['error']}")
                if "stderr" in result and result["stderr"]:
                    stderr_lines = result["stderr"].split("\n")[:5]  # First 5 lines
                    for line in stderr_lines:
                        if line.strip():
                            print(f"    {line}")

        print("=" * 60)

        return report

    def monitor_resources(self, interval: int = 5):
        """Monitor system resources during test execution."""
        # This would run in a separate thread to monitor resources
        # For now, we'll just log current resource usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()

        logger.info(
            f"Current resource usage: {memory_mb:.1f}MB RAM, {cpu_percent:.1f}% CPU"
        )


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive audio flow tests")

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "unit",
            "integration",
            "performance",
            "error_handling",
            "format_compatibility",
        ],
        default=None,
        help="Test categories to run (default: all except performance)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip performance tests)",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    parser.add_argument(
        "--output",
        default="comprehensive_test_report.json",
        help="Output file for test report",
    )

    parser.add_argument(
        "--working-dir", default=".", help="Working directory for tests"
    )

    args = parser.parse_args()

    # Change to working directory
    if args.working_dir != ".":
        import os

        os.chdir(args.working_dir)
        logger.info(f"Changed working directory to: {args.working_dir}")

    # Initialize test runner
    runner = TestSuiteRunner()

    # Run tests
    logger.info("Starting comprehensive audio test suite...")

    try:
        results = runner.run_comprehensive_suite(
            categories=args.categories, verbose=args.verbose, quick=args.quick
        )

        # Generate report
        report = runner.generate_report(args.output)

        # Exit with appropriate code
        failed_count = len([r for r in results.values() if r.get("exit_code") != 0])
        sys.exit(0 if failed_count == 0 else 1)

    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test run failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
