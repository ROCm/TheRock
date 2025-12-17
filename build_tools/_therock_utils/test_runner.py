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
        runner = TestRunner(component="rocwmma", test_type="full")
        runner.run_ctest(
            test_dir=Path(f"{THEROCK_BIN_DIR}/rocwmma"),
            parallel=8,
            timeout="3600"
        )
    """
    
    def __init__(self, component: str, test_type: str = "full"):
        """
        Initialize test runner
        
        Args:
            component: Name of the component being tested (e.g., "rocwmma", "hipblaslt")
            test_type: Type of test run (e.g., "full", "smoke", "regression")
        """
        self.component = component
        self.test_type = test_type
        self.logger = get_logger(__name__, component=component, operation="test")
        
    def run_gtest(
        self, 
        cmd: List[str], 
        cwd: Path, 
        env: Optional[Dict] = None,
        capture_output: bool = True
    ) -> int:
        """
        Run a gtest executable with unified logging
        
        Args:
            cmd: Command to execute (list of strings)
            cwd: Working directory
            env: Environment variables (optional)
            capture_output: Whether to capture and parse output
        
        Returns:
            Exit code (0 for success)
            
        Raises:
            subprocess.CalledProcessError: If test fails
        """
        self.logger.info(f"🧪 Running {self.component} GTest ({self.test_type})")
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
        test_dir: Path,
        parallel: int = 8,
        timeout: Optional[str] = None,
        tests_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict] = None
    ) -> int:
        """
        Run ctest with unified logging
        
        Args:
            test_dir: Directory containing tests (--test-dir)
            parallel: Number of parallel test jobs
            timeout: Per-test timeout in seconds
            tests_regex: Regex to filter tests to run
            exclude_regex: Regex to exclude tests
            extra_args: Additional ctest arguments
            cwd: Working directory (defaults to test_dir.parent)
            env: Environment variables
        
        Returns:
            Exit code (0 for success)
            
        Raises:
            subprocess.CalledProcessError: If test fails
        """
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
        
        self.logger.info(f"🧪 Running {self.component} CTest ({self.test_type})")
        self.logger.info(f"Command: {shlex.join(cmd)}")
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
        Parse gtest output to extract test results
        
        Example gtest output:
            [==========] Running 45 tests from 10 test suites.
            [  PASSED  ] 43 tests.
            [  FAILED  ] 2 tests, listed below:
            [  FAILED  ] TestSuite.TestName
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_tests": []
        }
        
        for line in output.splitlines():
            # [==========] Running 45 tests from 10 test suites.
            if match := re.search(r'\[==========\] Running (\d+) tests', line):
                results["total"] = int(match.group(1))
            
            # [  PASSED  ] 43 tests.
            elif match := re.search(r'\[  PASSED  \] (\d+) tests', line):
                results["passed"] = int(match.group(1))
            
            # [  FAILED  ] 2 tests, listed below:
            elif match := re.search(r'\[  FAILED  \] (\d+) tests', line):
                results["failed"] = int(match.group(1))
            
            # [  FAILED  ] TestSuite.TestName
            elif match := re.search(r'\[  FAILED  \] (.+)', line):
                if "tests, listed below" not in line:
                    results["failed_tests"].append(match.group(1))
        
        return results
    
    def _parse_ctest_output(self, output: str) -> Dict:
        """
        Parse ctest output to extract test results
        
        Example ctest output:
            Test project /path/to/tests
            Start 1: TestName1
            1/10 Test #1: TestName1 ........................   Passed    0.52 sec
            ...
            100% tests passed, 0 tests failed out of 10
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_tests": []
        }
        
        for line in output.splitlines():
            # 100% tests passed, 0 tests failed out of 10
            if match := re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', line):
                results["total"] = int(match.group(3))
                results["failed"] = int(match.group(2))
                results["passed"] = results["total"] - results["failed"]
            
            # Failed test lines
            elif "***Failed" in line or "***Timeout" in line:
                if match := re.search(r'Test #\d+: (.+?) \.*', line):
                    results["failed_tests"].append(match.group(1))
        
        return results
    
    def _log_test_results(self, results: Dict, returncode: int):
        """Log test results with appropriate log levels and GitHub annotations"""
        total = results.get("total", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        
        if total > 0:
            self.logger.info(
                f"Test Results: {passed}/{total} passed, {failed} failed",
                extra={
                    "test_total": total,
                    "test_passed": passed,
                    "test_failed": failed,
                    "component": self.component,
                    "test_type": self.test_type
                }
            )
        
        if returncode == 0:
            if total > 0:
                self.logger.info(f"✅ {self.component}: All {total} tests passed")
            else:
                self.logger.info(f"✅ {self.component}: Tests completed successfully")
        else:
            # Log each failed test as a separate error
            for failed_test in results.get("failed_tests", []):
                self.logger.error(f"Test failed: {failed_test}")
            
            if failed > 0:
                self.logger.error(
                    f"❌ {self.component}: {failed}/{total} tests failed"
                )
            else:
                self.logger.error(
                    f"❌ {self.component}: Tests failed with exit code {returncode}"
                )


def main():
    """Example usage"""
    print("This is a utility module. Import and use TestRunner class.")
    print("\nExample:")
    print("  from test_runner import TestRunner")
    print("  runner = TestRunner(component='mycomponent', test_type='full')")
    print("  runner.run_ctest(test_dir=Path('/path/to/tests'))")


if __name__ == '__main__':
    main()

