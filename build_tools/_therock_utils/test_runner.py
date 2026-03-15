#!/usr/bin/env python3
"""
Common Test Runner with Unified Logging

Provides unified logging support for gtest and ctest based test scripts.
Integrates with TheRock's unified logging framework to provide:
- GitHub Actions annotations
- Collapsible log groups
- Structured logging
- Test result parsing and reporting
"""

import subprocess
import shlex
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict
from logging_config import get_logger, configure_root_logger
import logging


class TestRunner:
    """
    Unified test runner with integrated logging for gtest/ctest
    
    Usage:
        # For GTest
        runner = TestRunner(component="rocROLLER", test_type="full", operation="gtest")
        runner.run_gtest(cmd=["./rocroller-tests"], cwd=Path("."))
        
        # For CTest
        runner = TestRunner(component="rocWMMA", test_type="full", operation="ctest")
        runner.run_ctest(test_dir=Path(f"{THEROCK_BIN_DIR}/rocwmma"), parallel=8)
    """
    
    def __init__(self, component: str, test_type: str = "full", operation: str = "test"):
        """
        Initialize test runner
        
        Args:
            component: Name of the component being tested (e.g., "rocwmma", "hipblaslt")
            test_type: Type of test run (e.g., "full", "smoke", "regression")
            operation: Operation type (e.g., "gtest", "ctest", "test")
        """
        self.component = component
        self.test_type = test_type
        # Use component name in logger for clear identification in logs
        logger_name = f"test_{component.lower().replace('-', '_')}"
        self.logger = get_logger(logger_name, component=component, operation=operation)
        
    def run_gtest(
        self, 
        cmd: Optional[List[str]] = None, 
        cwd: Optional[Path] = None, 
        env: Optional[Dict] = None,
        capture_output: bool = True,
        raw_output: Optional[str] = None
    ) -> int:
        """
        Run a gtest executable with unified logging
        
        Args:
            cmd: Command to execute (list of strings) - required if raw_output not provided
            cwd: Working directory - required if raw_output not provided
            env: Environment variables (optional)
            capture_output: Whether to capture and parse output (only used with cmd)
            raw_output: Raw GTest output string to parse directly (skips subprocess)
        
        Returns:
            Exit code (0 for success)
            
        Raises:
            subprocess.CalledProcessError: If test fails
        """
        self.logger.info(f"🧪 Running {self.component} GTest ({self.test_type})")
        
        # Handle raw_output mode (direct parsing without subprocess)
        if raw_output is not None:
            self.logger.info("Parsing raw GTest output")
            self.logger.info("")
            
            # Print raw output for visibility in logs
            self.logger.info("=== Raw GTest Output ===")
            for line in raw_output.splitlines():
                self.logger.info(line)
            self.logger.info("=== End Raw Output ===")
            self.logger.info("")
            
            try:
                with self.logger.timed_operation(f"{self.component}_gtest_parsing"):
                    # Parse gtest output directly
                    test_results = self._parse_gtest_output(raw_output)
                    exit_code = 1 if test_results.get("failed", 0) > 0 else 0
                    self._log_test_results(test_results, exit_code)
                    
                    return exit_code
            except Exception as e:
                self.logger.error(f"❌ Failed to parse GTest output: {e}", exc_info=True)
                raise
        
        # Regular subprocess mode
        if cmd is None or cwd is None:
            raise ValueError("cmd and cwd are required when raw_output is not provided")
        
        self.logger.info(f"Command: {shlex.join(cmd)}")
        self.logger.info(f"Working directory: {cwd}")
        
        if env:
            shard_info = []
            if "GTEST_SHARD_INDEX" in env:
                shard_info.append(f"shard {int(env['GTEST_SHARD_INDEX'])+1}/{env.get('GTEST_TOTAL_SHARDS', '?')}")
            if shard_info:
                self.logger.info(f"Sharding: {', '.join(shard_info)}")
        
        try:
            with self.logger.timed_operation(f"{self.component}_gtest_execution"):
                if capture_output:
                    result = subprocess.run(
                        cmd,
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    # Parse gtest output
                    test_results = self._parse_gtest_output(result.stdout)
                    self._log_test_results(test_results, result.returncode)
                    
                    # Log full output at DEBUG level
                    if result.stdout:
                        self.logger.debug("=== Test Output ===")
                        for line in result.stdout.splitlines():
                            self.logger.debug(line)
                    
                    if result.stderr:
                        for line in result.stderr.splitlines():
                            if line.strip():
                                self.logger.warning(f"stderr: {line}")
                    
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(
                            result.returncode, cmd, result.stdout, result.stderr
                        )
                    
                    return result.returncode
                else:
                    # Direct output to console (no parsing)
                    result = subprocess.run(
                        cmd,
                        cwd=cwd,
                        env=env,
                        check=True
                    )
                    self.logger.info(f"✅ {self.component} GTest completed successfully")
                    return result.returncode
                    
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"❌ {self.component} GTest failed (exit code {e.returncode})"
            )
            raise
    
    def run_ctest(
        self,
        test_dir: Optional[Path] = None,
        parallel: int = 8,
        timeout: Optional[str] = None,
        tests_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict] = None,
        raw_output: Optional[str] = None,
        cmd: Optional[List[str]] = None
    ) -> int:
        """
        Run ctest with unified logging
        
        Args:
            test_dir: Directory containing tests (--test-dir) - required if raw_output not provided
            parallel: Number of parallel test jobs
            timeout: Per-test timeout in seconds
            tests_regex: Regex to filter tests to run
            exclude_regex: Regex to exclude tests
            extra_args: Additional ctest arguments
            cwd: Working directory (defaults to test_dir.parent)
            env: Environment variables
            raw_output: Raw CTest output string to parse directly (skips subprocess)
            cmd: Custom command to execute (overrides test_dir and other params if provided)
        
        Returns:
            Exit code (0 for success)
            
        Raises:
            subprocess.CalledProcessError: If test fails
        """
        self.logger.info(f"🧪 Running {self.component} CTest ({self.test_type})")
        
        # Handle raw_output mode (direct parsing without subprocess)
        if raw_output is not None:
            self.logger.info("Parsing raw CTest output")
            self.logger.info("")
            
            # Print raw output for visibility in logs
            self.logger.info("=== Raw CTest Output ===")
            for line in raw_output.splitlines():
                self.logger.info(line)
            self.logger.info("=== End Raw Output ===")
            self.logger.info("")
            
            try:
                with self.logger.timed_operation(f"{self.component}_ctest_parsing"):
                    # Parse ctest output directly
                    test_results = self._parse_ctest_output(raw_output)
                    exit_code = 1 if test_results.get("failed", 0) > 0 else 0
                    self._log_test_results(test_results, exit_code)
                    
                    return exit_code
            except Exception as e:
                self.logger.error(f"❌ Failed to parse CTest output: {e}", exc_info=True)
                raise
        
        # Build command if not provided
        if cmd is None:
            if test_dir is None:
                raise ValueError("test_dir is required when raw_output and cmd are not provided")
            
            cmd = [
                "ctest",
                "--test-dir", str(test_dir),
                "--output-on-failure",
                "--parallel", str(parallel),
            ]
            
            if timeout:
                cmd.extend(["--timeout", timeout])
            
            if tests_regex:
                cmd.extend(["--tests-regex", tests_regex])
            
            if exclude_regex:
                cmd.extend(["--exclude-regex", exclude_regex])
            
            if extra_args:
                cmd.extend(extra_args)
            
            if cwd is None:
                cwd = test_dir.parent if test_dir.parent else Path.cwd()
        else:
            # Custom command provided
            if cwd is None:
                cwd = Path.cwd()
        
        self.logger.info(f"Command: {shlex.join(cmd)}")
        if test_dir:
            self.logger.info(f"Test directory: {test_dir}")
        self.logger.info(f"Working directory: {cwd}")
        self.logger.info(f"Parallelism: {parallel} jobs")
        if timeout:
            self.logger.info(f"Timeout: {timeout}s per test")
        
        try:
            with self.logger.timed_operation(f"{self.component}_ctest_execution"):
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                # Parse ctest output
                test_results = self._parse_ctest_output(result.stdout)
                self._log_test_results(test_results, result.returncode)
                
                # Log output
                if result.stdout:
                    self.logger.debug("=== CTest Output ===")
                    for line in result.stdout.splitlines():
                        self.logger.debug(line)
                
                if result.stderr:
                    for line in result.stderr.splitlines():
                        if line.strip():
                            self.logger.warning(f"stderr: {line}")
                
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd, result.stdout, result.stderr
                    )
                
                return result.returncode
                
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"❌ {self.component} CTest failed (exit code {e.returncode})"
            )
            raise
    
    def _parse_gtest_output(self, output: str) -> Dict:
        """
        Parse gtest output to extract test results and failure reasons
        
        Supports both formats:
        1. Standard GTest format:
            [==========] Running 45 tests from 10 test suites.
            [  PASSED  ] 43 tests.
            
        2. Logged format with PASSED/FAILED/SKIPPED keywords:
            2025-12-18 04:51:39 - INFO -    ✅ TestName: PASSED (50ms)
            2025-12-18 04:51:39 - ERROR -    ❌ TestName: FAILED
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_tests": [],
            "skipped_tests": [],
            "failure_details": {}  # Map of test_name -> failure reason
        }
        
        current_test = None
        failure_lines = []
        lines = output.splitlines()
        
        for i, line in enumerate(lines):
            # Track current running test
            if match := re.search(r'\[\s*RUN\s*\]\s+(.+)', line):
                current_test = match.group(1).strip()
                failure_lines = []
            
            # Capture test output (for all tests, saved only if test fails)
            elif current_test and not line.startswith('['):
                # Lines between [ RUN ] and test result markers are test output
                if line.strip() and not line.startswith('==='):
                    failure_lines.append(line.strip())
            
            # Try to extract test name and status from logged format first
            # Look for patterns like: "TestName: PASSED" or "❌ TestName: FAILED"
            if ": PASSED" in line:
                results["passed"] += 1
                # Extract test name (between "Running:" or emoji and ": PASSED")
                if match := re.search(r'(?:Running:|[✅❌⚠️]\s+)([A-Za-z0-9_\.]+):\s*PASSED', line):
                    results["total"] += 1
                current_test = None
                failure_lines = []
                    
            elif ": FAILED" in line:
                results["failed"] += 1
                # Extract test name
                if match := re.search(r'(?:Running:|[✅❌⚠️]\s+)([A-Za-z0-9_\.]+):\s*FAILED', line):
                    test_name = match.group(1).strip()
                    if test_name not in results["failed_tests"]:
                        results["failed_tests"].append(test_name)
                        results["total"] += 1
                current_test = None
                failure_lines = []
                        
            elif ": SKIPPED" in line or "SKIPPED" in line:
                # Extract test name
                if match := re.search(r'(?:Running:|[✅❌⚠️]\s+)([A-Za-z0-9_\.]+):\s*SKIPPED', line):
                    test_name = match.group(1).strip()
                    if test_name not in results["skipped_tests"]:
                        results["skipped_tests"].append(test_name)
                        results["skipped"] += 1
                        results["total"] += 1
                current_test = None
                failure_lines = []
            
            # Fallback to standard GTest format parsing
            # [==========] Running 45 tests from 10 test suites.
            elif match := re.search(r'\[==========\] Running (\d+) tests', line):
                results["total"] = int(match.group(1))
            
            # [  PASSED  ] 43 tests.
            elif match := re.search(r'\[  PASSED  \] (\d+) tests', line):
                results["passed"] = int(match.group(1))
            
            # [  FAILED  ] 2 tests, listed below:
            elif match := re.search(r'\[  FAILED  \] (\d+) tests?', line):
                results["failed"] = int(match.group(1))
            
            # [  SKIPPED ] 1 test, listed below:
            elif match := re.search(r'\[  SKIPPED \] (\d+) tests?', line):
                results["skipped"] = int(match.group(1))
            
            # [  FAILED  ] TestSuite.TestName (with timing or without)
            elif match := re.search(r'\[\s*FAILED\s*\]\s+(.+)', line):
                if "listed below" not in line:
                    test_name = match.group(1).strip()
                    # Remove timing info like "(50 ms)" if present
                    test_name = re.sub(r'\s*\(\d+\s*ms\)\s*$', '', test_name)
                    if test_name not in results["failed_tests"]:
                        results["failed_tests"].append(test_name)
                    
                    # Store failure details if we captured any
                    if failure_lines and test_name not in results["failure_details"]:
                        # Keep only the most relevant lines (first few lines usually contain the error)
                        failure_msg = "\n".join(failure_lines[:5])
                        results["failure_details"][test_name] = failure_msg
                    
                    current_test = None
                    failure_lines = []
            
            # [  SKIPPED ] TestSuite.TestName
            elif match := re.search(r'\[\s*SKIPPED\s*\]\s+(.+)', line):
                if "listed below" not in line:
                    test_name = match.group(1).strip()
                    # Remove timing info like "(0 ms)" if present
                    test_name = re.sub(r'\s*\(\d+\s*ms\)\s*$', '', test_name)
                    if test_name not in results["skipped_tests"]:
                        results["skipped_tests"].append(test_name)
                current_test = None
                failure_lines = []
        
        return results
    
    def _parse_ctest_output(self, output: str) -> Dict:
        """
        Parse ctest output to extract test results and failure reasons
        
        Example ctest output:
            Test project /path/to/tests
            Start 1: TestName1
            1/10 Test #1: TestName1 ........................   Passed    0.52 sec
            2/10 Test #2: TestName2 ........................***Skipped   0.00 sec
            3/10 Test #3: TestName3 ........................***Failed    1.23 sec
            ...
            100% tests passed, 0 tests failed out of 10
            
        With --output-on-failure, failed test output is shown between tests
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_tests": [],
            "skipped_tests": [],
            "failure_details": {}  # Map of test_name -> failure reason
        }
        
        lines = output.splitlines()
        current_failed_test = None
        failure_output = []
        
        for i, line in enumerate(lines):
            # 100% tests passed, 0 tests failed out of 10
            if match := re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', line):
                results["total"] = int(match.group(3))
                results["failed"] = int(match.group(2))
                results["passed"] = results["total"] - results["failed"] - results["skipped"]
            
            # Test result lines with status
            # Format: Test #N: TestName ....... Passed/Failed/Skipped  X.XX sec
            elif match := re.search(r'(\d+)/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*\*?\*?\*?(\w+)', line):
                test_name = match.group(2)
                status = match.group(3)
                
                if status == "Failed" or status == "Timeout":
                    results["failed_tests"].append(test_name)
                    current_failed_test = test_name
                    failure_output = []
                elif status == "Skipped" or status == "Disabled" or status == "NotRun":
                    results["skipped_tests"].append(test_name)
                    results["skipped"] += 1
                    current_failed_test = None
                else:
                    # Save captured failure output if we have any
                    if current_failed_test and failure_output:
                        failure_msg = "\n".join(failure_output[:10])  # Keep first 10 lines
                        results["failure_details"][current_failed_test] = failure_msg
                    current_failed_test = None
                    failure_output = []
            
            # Capture output lines for failed tests (lines between failed test and next test/summary)
            elif current_failed_test:
                # Skip empty lines, formatting lines, and "Start N:" lines
                if line.strip() and not line.startswith('===') and not line.startswith('---'):
                    # Skip test result lines and "Start N:" lines
                    if not re.match(r'^\s*\d+/\d+\s+Test\s+#\d+:', line) and not re.match(r'^\s*Start\s+\d+:', line):
                        failure_output.append(line.strip())
        
        # Save any remaining failure output
        if current_failed_test and failure_output:
            failure_msg = "\n".join(failure_output[:10])
            results["failure_details"][current_failed_test] = failure_msg
        
        return results
    
    def _log_test_results(self, results: Dict, returncode: int):
        """Log test results with detailed formatted output"""
        total = results.get("total", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        skipped = results.get("skipped", 0)
        failed_tests = results.get("failed_tests", [])
        skipped_tests = results.get("skipped_tests", [])
        failure_details = results.get("failure_details", {})
        
        # Calculate success rate
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Print header
        self.logger.info("📊 Test Results Summary")
        self.logger.info("=" * 60)
        
        # Print summary line with structured data
        self.logger.info(
            f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped",
            extra={
                "test_total": total,
                "test_passed": passed,
                "test_failed": failed,
                "test_skipped": skipped,
                "component": self.component,
                "test_type": self.test_type,
                "success_rate": f"{success_rate:.1f}%"
            }
        )
        
        # Print detailed breakdown
        self.logger.info(f"   Total Tests: {total}")
        self.logger.info(f"   ✅ Passed: {passed}")
        self.logger.info(f"   ❌ Failed: {failed}")
        self.logger.info(f"   ⚠️  Skipped: {skipped}")
        self.logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        # Log failed test names with failure details
        if failed_tests:
            self.logger.info("")
            self.logger.error(f"❌ {len(failed_tests)} test(s) failed:")
            for failed_test in failed_tests:
                self.logger.error(f"   - {failed_test}")
                # Show failure reason if available
                if failed_test in failure_details:
                    self.logger.error(f"     Reason:")
                    for detail_line in failure_details[failed_test].splitlines():
                        self.logger.error(f"       {detail_line}")
        
        # Log skipped test names
        if skipped_tests:
            self.logger.info("")
            self.logger.warning(f"⚠️  {len(skipped_tests)} test(s) skipped:")
            for skipped_test in skipped_tests:
                self.logger.warning(f"   - {skipped_test}")
        
        # Print footer
        self.logger.info("=" * 60)
        
        # Log final status
        if returncode == 0:
            self.logger.info("✅ All tests passed!")
        else:
            self.logger.error("❌ Some tests failed!")


def main():
    """Example usage"""
    print("This is a utility module. Import and use TestRunner class.")
    print("\nExample:")
    print("  from test_runner import TestRunner")
    print("  runner = TestRunner(component='mycomponent', test_type='full')")
    print("  runner.run_ctest(test_dir=Path('/path/to/tests'))")


if __name__ == '__main__':
    main()

