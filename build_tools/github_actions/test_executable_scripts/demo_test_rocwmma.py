#!/usr/bin/env python3
"""
Mock rocWMMA CTest with TestRunner Integration
===============================================
Demonstrates CTest integration with unified logging framework.
Generates raw CTest output and uses TestRunner to parse and log results.

Environment Variables:
- TEST_TYPE: smoke|quick|full (default: full)
- DEMO_FAILURES: Comma-separated test indices to fail (e.g., "3,7")
- DEMO_SKIPS: Comma-separated test indices to skip (e.g., "5")
- AMDGPU_FAMILIES: GPU families to test (e.g., "gfx942,gfx90a")
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
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "gfx942")

fail_indices = set(int(x.strip()) for x in DEMO_FAILURES.split(",") if x.strip())
skip_indices = set(int(x.strip()) for x in DEMO_SKIPS.split(",") if x.strip())
gpu_families = [f.strip() for f in AMDGPU_FAMILIES.split(",") if f.strip()]

# Mock test cases per GPU family
BASE_TESTS = [
    "gemm_float_test",
    "gemm_double_test",
    "gemm_half_test",
    "gemm_bfloat16_test",
    "dlrm_float_test",
    "dlrm_half_test",
    "attention_test",
    "layer_norm_test",
]

# Filter tests based on TEST_TYPE
if TEST_TYPE == "smoke":
    tests_per_gpu = ["gemm_float_test", "gemm_double_test"]
elif TEST_TYPE == "quick":
    tests_per_gpu = ["gemm_float_test", "gemm_double_test", "gemm_half_test", "dlrm_float_test"]
else:
    tests_per_gpu = BASE_TESTS

# Build full test list (each test x each GPU)
all_tests = []
for gpu in gpu_families:
    for test in tests_per_gpu:
        all_tests.append((test, gpu))

# Generate raw CTest format output
output_lines = []
output_lines.append("Test project /build/rocWMMA")
output_lines.append(f"    Start {1}: {all_tests[0][0]}")
output_lines.append("")

passed_count = 0
failed_tests = []
skipped_tests = []

for i, (test_name, gpu) in enumerate(all_tests, 1):
    full_test_name = f"{test_name}_{gpu}"
    duration = 2.5 + (i % 5) * 0.5
    
    output_lines.append(f"{i}/{len(all_tests)} Test #{i}: {test_name}")
    
    if i in fail_indices:
        output_lines.append(f"{i}/{len(all_tests)} Test #{i}: {test_name} ...***Failed {duration:.2f} sec")
        failed_tests.append(full_test_name)
        # Add failure output (simulating --output-on-failure)
        output_lines.append(f"Test failed with the following output:")
        output_lines.append(f"  GEMM kernel validation error on {gpu}")
        output_lines.append(f"  Expected output size: 1024x1024, Got: 1024x1023")
        output_lines.append(f"  Error code: -1 (Dimension mismatch)")
    elif i in skip_indices:
        output_lines.append(f"{i}/{len(all_tests)} Test #{i}: {test_name} ...***Skipped {duration:.2f} sec")
        skipped_tests.append(full_test_name)
    else:
        output_lines.append(f"{i}/{len(all_tests)} Test #{i}: {test_name} ...   Passed {duration:.2f} sec")
        passed_count += 1
    
    output_lines.append("")
    
    # Add "Start" line for next test if not last
    if i < len(all_tests):
        output_lines.append(f"    Start {i+1}: {all_tests[i][0]}")
        output_lines.append("")

# Summary
total_duration = sum(2.5 + (i % 5) * 0.5 for i in range(1, len(all_tests) + 1))
output_lines.append("")
output_lines.append(f"{100}% tests passed, {len(failed_tests)} tests failed out of {len(all_tests)}")
output_lines.append("")
if failed_tests or skipped_tests:
    output_lines.append("Label Time Summary:")
    output_lines.append(f"rocWMMA    = {total_duration:.2f} sec*proc ({len(all_tests)} tests)")
    output_lines.append("")

output_lines.append(f"Total Test time (real) = {total_duration:.2f} sec")
output_lines.append("")

if failed_tests:
    output_lines.append(f"The following tests FAILED:")
    for idx, test in enumerate(failed_tests, 1):
        # Find original test index
        orig_idx = next(i for i, (tn, gpu) in enumerate(all_tests, 1) if f"{tn}_{gpu}" == test)
        output_lines.append(f"\t{orig_idx} - {test.rsplit('_', 1)[0]} (Failed)")

if skipped_tests:
    output_lines.append(f"The following tests were SKIPPED:")
    for idx, test in enumerate(skipped_tests, 1):
        orig_idx = next(i for i, (tn, gpu) in enumerate(all_tests, 1) if f"{tn}_{gpu}" == test)
        output_lines.append(f"\t{orig_idx} - {test.rsplit('_', 1)[0]} (Skipped)")

output_lines.append(f"Errors while running CTest" if failed_tests else "")

raw_output = "\n".join(output_lines)

# Use TestRunner to parse and log the raw output
runner = TestRunner(component="rocWMMA", test_type=TEST_TYPE, operation="ctest")

try:
    exit_code = runner.run_ctest(raw_output=raw_output)
    sys.exit(exit_code)
except Exception as e:
    runner.logger.error(f"Failed to run demo: {e}", exc_info=True)
    sys.exit(1)
