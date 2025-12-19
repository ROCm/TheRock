#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test Results Capture Wrapper
=============================
Wrapper script to capture test output from CI jobs and parse with TestRunner.

Usage:
    # For GTest executables:
    python3 capture_test_results.py \
        --component rocBLAS \
        --test-type nightly \
        --framework gtest \
        --command ./build/rocblas-tests
    
    # For CTest:
    python3 capture_test_results.py \
        --component rocWMMA \
        --test-type nightly \
        --framework ctest \
        --cwd ./build/rocwmma \
        --command ctest --output-on-failure
"""

import argparse
import subprocess
import sys
from pathlib import Path
from test_runner import TestRunner


def capture_and_parse_tests(
    component: str,
    test_type: str,
    framework: str,
    command: str,
    cwd: Path = None,
    timeout: int = 600
) -> int:
    """
    Capture test output and parse with TestRunner.
    
    Parameters:
    -----------
    component : str
        Component name (e.g., rocBLAS, rocWMMA)
    test_type : str
        Test type (e.g., nightly, smoke, full)
    framework : str
        Test framework (gtest or ctest)
    command : str
        Command to execute
    cwd : Path, optional
        Working directory for command
    timeout : int
        Timeout in seconds (default: 600)
    
    Returns:
    --------
    int
        Exit code (0 for success, non-zero for failures)
    """
    print(f"Running {component} {framework} tests ({test_type})")
    print(f"Command: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    print()
    
    # Run the test command and capture output
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Combine stdout and stderr
        output = result.stdout + result.stderr
        
        print(f"Test command exited with code: {result.returncode}")
        print()
        
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Test command timed out after {timeout} seconds")
        output = (e.stdout or "") + (e.stderr or "")
        result_returncode = 124  # Standard timeout exit code
        
    except Exception as e:
        print(f"ERROR: Failed to run test command: {e}")
        return 1
    
    # Parse output with TestRunner
    try:
        runner = TestRunner(
            component=component,
            test_type=test_type,
            operation=framework
        )
        
        if framework.lower() == "gtest":
            exit_code = runner.run_gtest(raw_output=output)
        elif framework.lower() == "ctest":
            exit_code = runner.run_ctest(raw_output=output)
        else:
            print(f"ERROR: Unknown framework: {framework}")
            return 1
        
        print()
        print(f"Test results saved to: test_results_{component}_{test_type}.json")
        
        return exit_code
        
    except Exception as e:
        print(f"ERROR: Failed to parse test output: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(argv):
    parser = argparse.ArgumentParser(
        description="Capture and parse test results from CI jobs"
    )
    parser.add_argument(
        "--component",
        required=True,
        help="Component name (e.g., rocBLAS, rocWMMA, hipBLASLt)"
    )
    parser.add_argument(
        "--test-type",
        default="nightly",
        help="Test type (default: nightly)"
    )
    parser.add_argument(
        "--framework",
        required=True,
        choices=["gtest", "ctest"],
        help="Test framework (gtest or ctest)"
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Command to execute (e.g., './rocblas-tests' or 'ctest --output-on-failure')"
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help="Working directory for command (optional)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)"
    )
    
    args = parser.parse_args(argv)
    
    exit_code = capture_and_parse_tests(
        component=args.component,
        test_type=args.test_type,
        framework=args.framework,
        command=args.command,
        cwd=args.cwd,
        timeout=args.timeout
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])

