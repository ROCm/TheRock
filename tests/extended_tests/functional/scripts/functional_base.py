"""
Base class for functional tests with common functionality.

Provides test execution, result parsing, logging, and GitHub Actions integration
for functional correctness tests. Unlike benchmarks which measure performance,
functional tests verify correctness with pass/fail results.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, IO, List
from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Add build_tools/github_actions to path for github_actions_utils
sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "build_tools" / "github_actions")
)
from utils import TestClient, HardwareDetector
from utils.logger import log
from utils.exceptions import TestExecutionError, TestResultError
from github_actions_utils import gha_append_step_summary


class FunctionalBase:
    """Base class providing common functional test logic.

    Child classes must implement run_tests() and parse_results().

    Unlike benchmarks (which measure performance), functional tests verify
    correctness and produce pass/fail results without performance metrics.
    """

    def __init__(self, test_name: str, display_name: str = None):
        """Initialize functional test.

        Args:
            test_name: Internal test name (e.g., 'miopen_driver_conv')
            display_name: Display name for reports (e.g., 'MIOpen Driver Convolution')
        """
        self.test_name = test_name
        self.display_name = display_name or test_name

        # Environment variables
        self.therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
        self.artifact_run_id = os.getenv("ARTIFACT_RUN_ID")
        self.amdgpu_families = os.getenv("AMDGPU_FAMILIES")
        self.script_dir = Path(__file__).resolve().parent
        self.therock_dir = Path(__file__).resolve().parents[4]

        # Initialize test client (will be set in run())
        self.client = None

    def load_config(self, config_filename: str) -> Dict[str, Any]:
        """Load test configuration from JSON file.

        Args:
            config_filename: Name of JSON config file (e.g., 'miopen_driver_conv.json')

        Returns:
            Parsed JSON configuration dictionary

        Raises:
            TestExecutionError: If config file not found or invalid JSON
        """
        config_file = self.script_dir.parent / "configs" / config_filename

        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise TestExecutionError(
                f"Configuration file not found: {config_file}\n"
                f"Ensure {config_filename} exists in configs/ directory"
            )
        except json.JSONDecodeError as e:
            raise TestExecutionError(
                f"Invalid JSON in configuration file: {e}\n"
                f"Check JSON syntax in {config_filename}"
            )

    def get_gpu_architecture(self) -> str:
        """Detect GPU architecture using HardwareDetector.

        Returns:
            GPU architecture string (e.g., 'gfx942', 'gfx1100')

        Raises:
            TestExecutionError: If GPU detection fails or returns unknown
        """
        try:
            detector = HardwareDetector()
            gfx_id = detector.get_gpu_architecture()

            if not gfx_id or gfx_id == "unknown":
                raise TestExecutionError(
                    "Could not detect GPU architecture.\n"
                    "Ensure ROCm drivers are installed and GPU is accessible."
                )

            log.info(f"Detected GPU architecture: {gfx_id}")
            return gfx_id

        except TestExecutionError:
            # Re-raise TestExecutionError as-is
            raise
        except Exception as e:
            raise TestExecutionError(
                f"Failed to detect GPU architecture: {e}\n"
                "Ensure ROCm drivers are installed and GPU is accessible."
            ) from e

    def execute_command(
        self,
        cmd: List[str],
        cwd: Path = None,
        env: Dict[str, str] = None,
        log_file_handle: IO = None,
    ) -> int:
        """Execute a command and stream output.

        Args:
            cmd: Command list to execute
            cwd: Working directory (default: self.therock_dir)
            env: Optional environment variables to set
            log_file_handle: Optional file handle to write output

        Returns:
            Exit code from the command
        """
        work_dir = cwd or self.therock_dir
        log.info(f"++ Exec [{work_dir}]$ {shlex.join(cmd)}")
        if log_file_handle:
            log_file_handle.write(f"{shlex.join(cmd)}\n")

        # Merge custom env with current environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        for line in process.stdout:
            log.info(line.strip())
            if log_file_handle:
                log_file_handle.write(f"{line}")

        process.wait()
        return process.returncode

    def create_test_result(
        self,
        test_name: str,
        subtest_name: str,
        status: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a standardized functional test result dictionary.

        Args:
            test_name: Test name
            subtest_name: Specific test/suite identifier
            status: Test status ('PASS', 'FAIL', 'ERROR', 'SKIP')
            **kwargs: Additional test-specific parameters (pass_count, fail_count, etc.)

        Returns:
            Dict[str, Any]: Test result dictionary
        """
        test_config = {
            "test_name": test_name,
            "sub_test_name": subtest_name,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "environment_dependencies": [],
        }

        # Add any additional kwargs to test_config
        for key, value in kwargs.items():
            test_config[key] = value

        return {
            "test_name": test_name,
            "subtest": subtest_name,
            "status": status,
            "test_config": test_config,
            **kwargs,  # Include pass_count, fail_count, etc. at top level too
        }

    def calculate_statistics(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate test statistics from results.

        Args:
            test_results: List of test result dictionaries with 'status' key

        Returns:
            Dictionary with detailed statistics including all status types
        """
        passed = sum(1 for r in test_results if r.get("status") == "PASS")
        failed = sum(1 for r in test_results if r.get("status") == "FAIL")
        error = sum(1 for r in test_results if r.get("status") == "ERROR")
        skipped = sum(1 for r in test_results if r.get("status") == "SKIP")

        # Overall status: PASS only if no failures/errors
        overall_status = "PASS" if (failed == 0 and error == 0) else "FAIL"

        return {
            "passed": passed,
            "failed": failed,
            "error": error,
            "skipped": skipped,
            "total": len(test_results),
            "overall_status": overall_status,
        }

    def create_summary_table(
        self, stats: Dict[str, Any], num_suites: int
    ) -> PrettyTable:
        """Create overall summary table with all statistics.

        Args:
            stats: Test statistics dictionary
            num_suites: Number of test suites

        Returns:
            PrettyTable with summary statistics
        """
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

        # Set consistent column width for uniform appearance
        for field in summary_table.field_names:
            summary_table.min_width[field] = 17

        summary_table.add_row(
            [
                num_suites,
                stats["total"],
                stats["passed"],
                stats["failed"],
                stats["error"],
                stats["skipped"],
                stats["overall_status"],
            ]
        )

        return summary_table

    def upload_results(
        self, test_results: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> bool:
        """Upload results to API and save locally.

        Args:
            test_results: List of test result dictionaries
            stats: Test statistics dictionary

        Returns:
            True if upload successful, False otherwise
        """
        log.info("Uploading Functional Tests Results to API")
        success = self.client.upload_results(
            test_name=f"{self.test_name}_functional",
            test_results=test_results,
            test_status=stats["overall_status"],
            test_metadata={
                "artifact_run_id": self.artifact_run_id,
                "amdgpu_families": self.amdgpu_families,
                "test_name": self.test_name,
                "total_tests": stats["total"],
                "passed_tests": stats["passed"],
                "failed_tests": stats["failed"],
                "error_tests": stats["error"],
                "skipped_tests": stats["skipped"],
            },
            save_local=True,
            output_dir=str(self.script_dir.parent / "results"),
        )

        if success:
            log.info("Results uploaded successfully")
        else:
            log.info("Results saved locally only (API upload disabled or failed)")

        return success

    def run(self) -> None:
        """Execute functional test workflow.

        Raises:
            TestExecutionError: If test execution encounters errors (missing files, etc.)
            TestResultError: If tests run successfully but results show failures

        Note:
            On success, returns normally (exit code 0)
            On failure, raises exception (exit code 1)
        """
        log.info(f"{self.display_name} - Starting Functional Test")

        # Initialize test client and print system info
        self.client = TestClient(auto_detect=True)
        self.client.print_system_summary()

        # Run tests (implemented by child class)
        self.run_tests()

        # Parse results (implemented by child class)
        test_results, detailed_table, num_suites = self.parse_results()

        # Validate test results structure
        if not test_results:
            raise TestExecutionError(
                "No test results generated - parse_results() returned empty list\n"
                "Check if tests executed successfully and results were saved to file"
            )

        # Calculate statistics
        stats = self.calculate_statistics(test_results)

        # Create summary table
        summary_table = self.create_summary_table(stats, num_suites)

        # Display results
        log.info("DETAILED RESULTS")
        log.info(f"\n{detailed_table}")

        log.info("\nSUMMARY")
        log.info(f"\n{summary_table}")
        log.info(f"\nFinal Status: {stats['overall_status']}")

        # Upload results (optional, may not be available in all environments)
        try:
            self.upload_results(test_results, stats)
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


def run_functional_test_main(test_instance):
    """Run functional test.

    Raises exceptions on failure, returns normally on success.
    This is the Pythonic way - let Python set exit codes automatically:
    - Success: Returns normally → exit code 0
    - Execution Error: Raises TestExecutionError → exit code 1
    - Result Failure: Raises TestResultError → exit code 1
    """
    test_instance.run()
