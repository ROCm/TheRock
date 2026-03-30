# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for functional tests with common functionality."""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.logger import log
from utils.exceptions import TestExecutionError, TestResultError
from utils.extended_test_base import ExtendedTestBase, gha_append_step_summary


class FunctionalBase(ExtendedTestBase):
    """Base class providing common functional test logic.

    Inherits shared infrastructure from ExtendedTestBase (execute_command,
    create_test_result, calculate_statistics, upload_results, etc.).

    Child classes must implement run_tests() and parse_results().
    """

    def __init__(self, test_name: str, display_name: str = None):
        """Initialize functional test.

        Args:
            test_name: Internal test name (e.g., 'miopen_driver_conv')
            display_name: Display name for reports (e.g., 'MIOpen Driver Convolution')
        """
        super().__init__(test_name, display_name or test_name)
        self.script_dir = Path(__file__).resolve().parent

    def execute_command_with_output(
        self, cmd: List[str], cwd: Path = None, env: Dict[str, str] = None
    ) -> Tuple[int, str]:
        """Execute a command and return its exit code and captured output.

        Unlike ExtendedTestBase.execute_command (which returns only the exit
        code), this variant also captures and returns stdout/stderr as a single
        string.  Useful when the caller needs to parse or inspect the output.
        """
        work_dir = cwd or self.therock_dir
        log.info(f"++ Exec [{work_dir}]$ {shlex.join(cmd)}")

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
        )

        output_lines = []
        for line in process.stdout:
            line_text = line.rstrip()
            log.info(line_text)
            output_lines.append(line_text)

        process.wait()
        return process.returncode, "\n".join(output_lines)

    def get_rocm_env_with_path(self) -> Dict[str, str]:
        """Get environment with ROCm libraries on LD_LIBRARY_PATH and ROCm
        tool directories on PATH.

        Extends get_rocm_env() by prepending standard ROCm bin directories
        (hipcc, llvm tools, etc.) to PATH and setting HIP_PLATFORM=amd.
        """
        env = self.get_rocm_env()
        extra_dirs = [
            str(self.rocm_path / "bin"),
            str(self.rocm_path / "llvm" / "bin"),
        ]
        existing_path = env.get("PATH", "")
        env["PATH"] = ":".join(d for d in extra_dirs + [existing_path] if d)
        env["HIP_PLATFORM"] = "amd"
        return env

    def create_result_tables(
        self, test_results: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> tuple:
        """Create detailed and summary result tables."""
        # Build detailed table and count suites
        detailed_table = PrettyTable()
        detailed_table.field_names = ["TestSuite", "TestCase", "Status"]

        suites = set()
        for result in test_results:
            suite = result.get(
                "suite", result.get("test_config", {}).get("suite", "unknown")
            )
            subtest = result.get("subtest", "unknown")
            status = result.get("status", "FAIL")

            suites.add(suite)
            detailed_table.add_row([suite, subtest, status])

        # Build summary table
        summary_table = PrettyTable()
        summary_table.field_names = [
            "Total TestSuites",
            "Total TestCases",
            "Passed",
            "Failed",
            "Errored",
            "Skipped",
            "Final Result",
        ]

        for field in summary_table.field_names:
            summary_table.min_width[field] = 17

        summary_table.add_row(
            [
                len(suites),
                stats["total"],
                stats["passed"],
                stats["failed"],
                stats["error"],
                stats["skipped"],
                stats["overall_status"],
            ]
        )

        return detailed_table, summary_table

    def run(self) -> None:
        """Execute functional test workflow.

        Raises:
            TestExecutionError: If test execution encounters errors
            TestResultError: If tests run but results show failures
        """
        log.info(f"{self.display_name} - Starting Functional Test")

        # Run tests (implemented by child class)
        self.run_tests()

        # Parse results (implemented by child class)
        test_results = self.parse_results()

        # Validate test results structure
        if not test_results:
            raise TestExecutionError(
                "No test results generated - parse_results() returned empty list\n"
                "Check if tests executed successfully and results were saved to file"
            )

        # Generate statistics and tables
        stats = self.calculate_statistics(test_results)
        detailed_table, summary_table = self.create_result_tables(test_results, stats)

        # Display results
        log.info("DETAILED RESULTS")
        log.info(f"\n{detailed_table}")

        log.info("\nSUMMARY")
        log.info(f"\n{summary_table}")
        log.info(f"\nFinal Status: {stats['overall_status']}")

        # Upload results
        try:
            self.upload_results(
                test_results=test_results,
                stats=stats,
                test_type="functional",
                output_dir=str(self.script_dir.parent / "results"),
                extra_metadata={
                    "total_tests": stats["total"],
                    "passed_tests": stats["passed"],
                    "failed_tests": stats["failed"],
                    "error_tests": stats["error"],
                    "skipped_tests": stats["skipped"],
                },
            )
        except Exception as e:
            log.warning(f"Could not upload results: {e}")

        # Write to GitHub Actions step summary
        try:
            gha_append_step_summary(
                f"## {self.display_name} - Functional Test Results\n\n"
                f"```\n{summary_table}\n```\n"
            )
        except Exception as e:
            log.error(f"Could not write GitHub Actions summary: {e}")

        # Raise exception if tests failed
        if stats["overall_status"] != "PASS":
            failed = stats["failed"]
            errored = stats["error"]
            total = stats["total"]
            raise TestResultError(
                f"Test suite completed with failures: "
                f"{failed} failed, {errored} errors out of {total} total tests"
            )


def run_functional_main(test_instance):
    """Run functional test.

    Raises exceptions on failure, returns normally on success.
    - Success: Returns normally → exit code 0
    - Execution Error: Raises TestExecutionError → exit code 1
    - Result Failure: Raises TestResultError → exit code 1
    """
    test_instance.run()
