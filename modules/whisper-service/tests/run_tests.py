#!/usr/bin/env python3
"""
Test Runner for WebSocket Server Testing Suite

Comprehensive test runner for unit tests, integration tests, and stress tests.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import pytest

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("test_results.log")],
    )


def run_unit_tests(verbose: bool = False) -> int:
    """Run unit tests"""
    print("🔬 Running Unit Tests...")

    args = [
        "test_unit.py",
        "-v" if verbose else "",
        "--tb=short",
        "--durations=10",
        "--junit-xml=test_results_unit.xml",
    ]

    return pytest.main([arg for arg in args if arg])


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests"""
    print("🔗 Running Integration Tests...")

    args = [
        "test_integration.py",
        "-v" if verbose else "",
        "--tb=short",
        "--durations=10",
        "--junit-xml=test_results_integration.xml",
    ]

    return pytest.main([arg for arg in args if arg])


def run_stress_tests(verbose: bool = False) -> int:
    """Run stress tests"""
    print("⚡ Running Stress Tests...")

    args = [
        "test_stress.py",
        "-v" if verbose else "",
        "--tb=short",
        "--durations=10",
        "-m",
        "stress",
        "--junit-xml=test_results_stress.xml",
    ]

    return pytest.main([arg for arg in args if arg])


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance tests"""
    print("📊 Running Performance Tests...")

    args = [
        "test_integration.py::TestPerformanceIntegration",
        "test_stress.py::TestPerformanceBenchmarks",
        "-v" if verbose else "",
        "--tb=short",
        "--durations=10",
        "--junit-xml=test_results_performance.xml",
    ]

    return pytest.main([arg for arg in args if arg])


def run_all_tests(verbose: bool = False) -> dict[str, int]:
    """Run all test suites"""
    print("🚀 Running Complete Test Suite...")

    results = {}

    # Run test suites in order
    test_suites = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("Stress Tests", run_stress_tests),
    ]

    for suite_name, test_function in test_suites:
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")

        start_time = time.time()
        result = test_function(verbose)
        duration = time.time() - start_time

        results[suite_name] = {
            "exit_code": result,
            "duration": duration,
            "status": "PASSED" if result == 0 else "FAILED",
        }

        print(f"\n{suite_name} completed in {duration:.2f}s - {results[suite_name]['status']}")

    return results


def print_summary(results: dict[str, Any]):
    """Print test results summary"""
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")

    total_duration = sum(r["duration"] for r in results.values())
    passed_count = sum(1 for r in results.values() if r["exit_code"] == 0)
    total_count = len(results)

    for suite_name, result in results.items():
        status_emoji = "✅" if result["exit_code"] == 0 else "❌"
        print(f"{status_emoji} {suite_name:<20} {result['status']:<8} ({result['duration']:.2f}s)")

    print("\n📊 Overall Results:")
    print(f"   Total Suites: {total_count}")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {total_count - passed_count}")
    print(f"   Success Rate: {passed_count/total_count*100:.1f}%")
    print(f"   Total Duration: {total_duration:.2f}s")

    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready for production.")
    else:
        print("\n⚠️  Some tests failed. Please review the results before deploying.")


def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking test dependencies...")

    required_packages = [
        "pytest",
        "pytest-asyncio",
        "websockets",
        "aiohttp",
        "numpy",
        "soundfile",
        "redis",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies available")
    return True


def check_services():
    """Check if required services are running"""
    print("🔍 Checking required services...")

    import socket

    services = [
        ("WebSocket Server", "localhost", 5001),
        ("Redis", "localhost", 6379),
    ]

    available_services = []

    for service_name, host, port in services:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"   ✅ {service_name} ({host}:{port})")
                available_services.append(service_name)
            else:
                print(f"   ❌ {service_name} ({host}:{port}) - not reachable")
        except Exception as e:
            print(f"   ❌ {service_name} ({host}:{port}) - error: {e}")

    if len(available_services) < len(services):
        print("\n⚠️  Some services are not available. Tests may fail.")
        print("Make sure the WebSocket server is running with:")
        print("docker-compose -f docker-compose.dev.yml up -d")
        return False

    print("✅ All services available")
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="WebSocket Server Test Runner")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "stress", "performance", "all"],
        default="all",
        nargs="?",
        help="Type of tests to run",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip dependency and service checks"
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip test summary")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("🧪 WebSocket Server Test Runner")
    print("================================")

    # Pre-flight checks
    if not args.skip_checks:
        print("\n📋 Running pre-flight checks...")

        if not check_dependencies():
            print("❌ Dependency check failed. Exiting.")
            return 1

        if not check_services():
            print("⚠️  Service check failed. Some tests may fail.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                print("Exiting.")
                return 1

    # Change to test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    # Run tests based on selection
    if args.test_type == "unit":
        exit_code = run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        exit_code = run_integration_tests(args.verbose)
    elif args.test_type == "stress":
        exit_code = run_stress_tests(args.verbose)
    elif args.test_type == "performance":
        exit_code = run_performance_tests(args.verbose)
    elif args.test_type == "all":
        results = run_all_tests(args.verbose)
        if not args.no_summary:
            print_summary(results)
        exit_code = max(r["exit_code"] for r in results.values())

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
