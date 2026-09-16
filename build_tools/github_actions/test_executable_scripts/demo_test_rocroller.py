#!/usr/bin/env python3
"""
Mock rocROLLER Test with TestRunner Integration
================================================
Demonstrates GTest integration with unified logging framework.
Generates raw GTest output and uses TestRunner to parse and log results.

Environment Variables:
- TEST_TYPE: smoke|quick|full (default: full)
- DEMO_FAILURES: Comma-separated test indices to fail (e.g., "3,7")
- DEMO_SKIPS: Comma-separated test indices to skip (e.g., "5")
"""

import os
import sys
import time
from pathlib import Path

# Add _therock_utils to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_therock_utils"))

from test_runner import TestRunner

# Note: Logging is auto-configured when TestRunner is imported (defaults to INFO level)

# Parse environment variables
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()
DEMO_FAILURES = os.getenv("DEMO_FAILURES", "")
DEMO_SKIPS = os.getenv("DEMO_SKIPS", "")

fail_indices = set(int(x.strip()) for x in DEMO_FAILURES.split(",") if x.strip())
skip_indices = set(int(x.strip()) for x in DEMO_SKIPS.split(",") if x.strip())

# Mock test cases
ALL_TESTS = [
    "ErrorFixtureDeathTest.NullPointer",
    "ErrorFixtureDeathTest.InvalidArgument",
    "ArgumentLoaderTest.BasicLoad",
    "ArgumentLoaderTest.ComplexLoad",
    "AssemblerTest.SimpleAssembly",
    "AssemblerTest.ComplexAssembly",
    "ControlGraphTest.BasicGraph",
    "ControlGraphTest.NestedGraph",
    "CommandTest.SingleCommand",
    "CommandTest.MultipleCommands",
    "ComponentTest.Initialization",
    "ComponentTest.Configuration",
    "MatrixMultiplyTest.SmallMatrix",
    "MatrixMultiplyTest.LargeMatrix",
    "PerformanceTest.Benchmark",
    "IntegrationTest.EndToEnd",
]

# Filter tests based on TEST_TYPE
if TEST_TYPE == "smoke":
    smoke_patterns = ["ErrorFixtureDeathTest", "ArgumentLoaderTest", "AssemblerTest", 
                     "ControlGraphTest", "CommandTest", "ComponentTest"]
    tests_to_run = [t for t in ALL_TESTS if any(p in t for p in smoke_patterns)]
elif TEST_TYPE == "quick":
    tests_to_run = [t for t in ALL_TESTS if "Small" in t or "Basic" in t or "Simple" in t]
else:
    tests_to_run = ALL_TESTS

# Generate raw GTest format output
output_lines = []
output_lines.append(f"[==========] Running {len(tests_to_run)} tests from {len(set(t.split('.')[0] for t in tests_to_run))} test suites.")
output_lines.append("[----------] Global test environment set-up.")

passed_count = 0
failed_tests = []
skipped_tests = []
test_suite = None

for i, test_name in enumerate(tests_to_run, 1):
    suite_name, case_name = test_name.split('.')
    
    # Print test suite header if changed
    if suite_name != test_suite:
        if test_suite is not None:
            output_lines.append(f"[----------] {suite_count} tests from {test_suite}")
        test_suite = suite_name
        suite_count = sum(1 for t in tests_to_run if t.startswith(suite_name))
        output_lines.append(f"[----------] {suite_count} tests from {suite_name}")
    
    output_lines.append(f"[ RUN      ] {test_name}")
    
    if i in fail_indices:
        output_lines.append(f"{test_name}:42: Failure")
        output_lines.append(f"Expected: (result) == (42), actual: 41")
        output_lines.append(f"[  FAILED  ] {test_name} (50 ms)")
        failed_tests.append(test_name)
    elif i in skip_indices:
        output_lines.append(f"[  SKIPPED ] {test_name} (0 ms)")
        skipped_tests.append(test_name)
    else:
        output_lines.append(f"[       OK ] {test_name} (50 ms)")
        passed_count += 1

if test_suite:
    output_lines.append(f"[----------] {suite_count} tests from {test_suite}")

output_lines.append("[----------] Global test environment tear-down")
output_lines.append(f"[==========] {len(tests_to_run)} tests from {len(set(t.split('.')[0] for t in tests_to_run))} test suites ran.")
output_lines.append(f"[  PASSED  ] {passed_count} tests.")

if failed_tests:
    output_lines.append(f"[  FAILED  ] {len(failed_tests)} tests, listed below:")
    for test in failed_tests:
        output_lines.append(f"[  FAILED  ] {test}")
    output_lines.append("")
    output_lines.append(f" {len(failed_tests)} FAILED TEST{'S' if len(failed_tests) > 1 else ''}")

if skipped_tests:
    output_lines.append(f"[  SKIPPED ] {len(skipped_tests)} tests, listed below:")
    for test in skipped_tests:
        output_lines.append(f"[  SKIPPED ] {test}")

raw_output = "\n".join(output_lines)

# Use TestRunner to parse and log the raw output
runner = TestRunner(component="rocROLLER", test_type=TEST_TYPE, operation="gtest")

try:
    exit_code = runner.run_gtest(raw_output=raw_output)
    sys.exit(exit_code)
except Exception as e:
    runner.logger.error(f"Failed to run demo: {e}", exc_info=True)
    sys.exit(1)
