#!/usr/bin/env python3
"""
Sample GTest Runner - Demonstrates logging framework usage with GTest
"""

import time
import logging
import subprocess
from logging_config import get_logger, configure_root_logger


def create_mock_gtest_output():
    """
    Creates mock GTest output to simulate a real test run
    This simulates what a real GTest binary would output
    """
    return """[==========] Running 15 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 5 tests from MathTest
[ RUN      ] MathTest.Addition
[       OK ] MathTest.Addition (0 ms)
[ RUN      ] MathTest.Subtraction
[       OK ] MathTest.Subtraction (1 ms)
[ RUN      ] MathTest.Multiplication
[       OK ] MathTest.Multiplication (0 ms)
[ RUN      ] MathTest.Division
[       OK ] MathTest.Division (1 ms)
[ RUN      ] MathTest.DivisionByZero
[  FAILED  ] MathTest.DivisionByZero (0 ms)
[----------] 5 tests from MathTest (2 ms total)

[----------] 6 tests from StringTest
[ RUN      ] StringTest.Length
[       OK ] StringTest.Length (0 ms)
[ RUN      ] StringTest.Concatenation
[       OK ] StringTest.Concatenation (0 ms)
[ RUN      ] StringTest.Comparison
[       OK ] StringTest.Comparison (1 ms)
[ RUN      ] StringTest.EmptyString
[       OK ] StringTest.EmptyString (0 ms)
[ RUN      ] StringTest.Unicode
[       OK ] StringTest.Unicode (1 ms)
[ RUN      ] StringTest.LargeString
[       OK ] StringTest.LargeString (2 ms)
[----------] 6 tests from StringTest (4 ms total)

[----------] 4 tests from GPUTest
[ RUN      ] GPUTest.DeviceQuery
[       OK ] GPUTest.DeviceQuery (45 ms)
[ RUN      ] GPUTest.MemoryAllocation
[       OK ] GPUTest.MemoryAllocation (12 ms)
[ RUN      ] GPUTest.KernelExecution
[       OK ] GPUTest.KernelExecution (89 ms)
[ RUN      ] GPUTest.DataTransfer
[       OK ] GPUTest.DataTransfer (23 ms)
[----------] 4 tests from GPUTest (169 ms total)

[----------] Global test environment tear-down
[==========] 15 tests from 3 test suites ran. (175 ms total)
[  PASSED  ] 14 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] MathTest.DivisionByZero

 1 FAILED TEST
"""


def parse_gtest_output(output):
    """
    Parse GTest output and extract test results
    
    Demonstrates:
    - Parsing structured test output
    - Extracting key metrics
    - Structured logging with parsed data
    """
    logger = get_logger(__name__, component="GTestRunner", operation="parse")
    
    logger.info("Parsing GTest output")
    
    # Simple parsing logic
    lines = output.split('\n')
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    test_suites = 0
    duration_ms = 0
    
    for line in lines:
        if 'tests from' in line and 'test suites' in line:
            # Extract total tests
            parts = line.split()
            if 'Running' in line:
                total_tests = int(parts[parts.index('Running') + 1])
                test_suites = int(parts[parts.index('from') + 1])
        elif '[  PASSED  ]' in line and 'tests' in line:
            parts = line.split()
            passed_tests = int(parts[parts.index('PASSED') + 2])
        elif '[  FAILED  ]' in line and 'test' in line:
            parts = line.split()
            failed_tests = int(parts[parts.index('FAILED') + 2])
        elif 'ms total)' in line and '==========' in line:
            # Extract total duration
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'ms' and i > 0:
                    duration_ms = int(parts[i-1].strip('('))
    
    results = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "test_suites": test_suites,
        "duration_ms": duration_ms,
        "success_rate": f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
    }
    
    logger.info("GTest results parsed", extra=results)
    
    return results


def run_gtest_suite(test_binary, test_filter=None):
    """
    Run a GTest binary with logging
    
    Demonstrates:
    - timed_operation: Automatic timing for test execution
    - Structured logging with test metrics
    - Output parsing and result tracking
    """
    logger = get_logger(__name__, component="GTestRunner", operation="execute")
    
    logger.info(f"Starting GTest execution", extra={
        "test_binary": test_binary,
        "test_filter": test_filter or "all"
    })
    
    # Simulate running GTest
    with logger.timed_operation(f"Execute {test_binary}"):
        logger.info(f"🧪 Running test binary: {test_binary}", extra={
            "test_binary": test_binary,
            "status": "running"
        })
        
        # In a real scenario, this would run: subprocess.run([test_binary, ...])
        # For demo purposes, we'll use mock output
        time.sleep(1)  # Simulate test execution
        
        output = create_mock_gtest_output()
        
        # Parse results
        results = parse_gtest_output(output)
        
        # Log results with structured data
        if results["failed_tests"] == 0:
            logger.info(f"✅ All tests passed!", extra=results)
        else:
            logger.warning(f"⚠️  Some tests failed", extra=results)
    
    return results


def run_multiple_test_suites():
    """
    Run multiple test suites with result aggregation
    
    Demonstrates:
    - Running multiple test components
    - Aggregating results
    - Performance tracking across suites
    """
    logger = get_logger(__name__, component="GTestRunner", operation="suite")
    
    test_suites = [
        "rocm-core-tests",
        "hip-runtime-tests",
        "rocblas-tests"
    ]
    
    logger.info(f"Running {len(test_suites)} test suites", extra={
        "suite_count": len(test_suites),
        "operation": "test_all"
    })
    
    all_results = []
    
    for suite in test_suites:
        with logger.timed_operation(f"Suite: {suite}"):
            results = run_gtest_suite(suite)
            all_results.append({
                "suite": suite,
                **results
            })
    
    # Aggregate results
    total_tests = sum(r["total_tests"] for r in all_results)
    total_passed = sum(r["passed_tests"] for r in all_results)
    total_failed = sum(r["failed_tests"] for r in all_results)
    total_duration = sum(r["duration_ms"] for r in all_results)
    
    aggregate = {
        "total_suites": len(test_suites),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_duration_ms": total_duration,
        "overall_success_rate": f"{(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "0%"
    }
    
    logger.info("All test suites completed", extra=aggregate)
    
    return aggregate


def demonstrate_test_filtering():
    """
    Demonstrate running tests with filters
    
    Demonstrates:
    - Test filtering (common GTest pattern)
    - Conditional test execution
    - Different test categories
    """
    logger = get_logger(__name__, component="GTestRunner", operation="filter")
    
    logger.info("Demonstrating test filtering")
    
    filters = [
        ("Smoke tests", "*Smoke*"),
        ("GPU tests only", "GPUTest.*"),
        ("Quick tests", "*Quick*:*Fast*")
    ]
    
    for filter_name, filter_pattern in filters:
        logger.info(f"🔍 Running filtered tests: {filter_name}", extra={
            "filter_name": filter_name,
            "filter_pattern": filter_pattern
        })
        
        # Simulate filtered test run
        with logger.timed_operation(f"Filter: {filter_name}"):
            time.sleep(0.5)
            logger.info(f"Filtered tests completed: {filter_name}", extra={
                "filter": filter_pattern,
                "status": "completed"
            })


def main():
    """Main demo function"""
    # Initialize logging with DEBUG level to see timed_operation's automatic logs
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 3: GTest Runner with Logging")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger(__name__, component="GTestRunner")
    logger.info("GTest Runner Demo Started")
    logger.debug("🔧 Logging initialized at DEBUG level - automatic timing enabled")
    
    # Run demos
    with logger.timed_operation("Complete GTest Demo"):
        # Demo 1: Single test suite
        logger.info("Demo 1: Running single test suite")
        run_gtest_suite("sample-gtest-binary")
        
        print()  # Spacing
        
        # Demo 2: Multiple test suites
        logger.info("Demo 2: Running multiple test suites")
        results = run_multiple_test_suites()
        
        print()  # Spacing
        
        # Demo 3: Test filtering
        logger.info("Demo 3: Test filtering demonstration")
        demonstrate_test_filtering()
    
    logger.info("GTest Runner Demo Completed", extra={
        "demo_sections": 3,
        "status": "success"
    })
    
    print("\n" + "="*60)
    print("  Key Features Demonstrated:")
    print("  - GTest execution with unified logging")
    print("  - Test result parsing and structured logging")
    print("  - Performance timing across test suites")
    print("  - Test filtering patterns")
    print("  - Automatic duration tracking")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()


