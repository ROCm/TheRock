"""Base class for benchmark tests with common functionality."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, IO
from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))  # benchmarks/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # github_actions/
from utils import BenchmarkClient, HardwareDetector
from utils.logger import log
from utils.exceptions import TestExecutionError, TestResultError
from github_actions_utils import gha_append_step_summary


class BenchmarkBase:
    """Base class providing common benchmark logic.

    Child classes must implement run_benchmarks() and parse_results().
    """

    def __init__(self, benchmark_name: str, display_name: str = None):
        """Initialize benchmark test.

        Args:
            benchmark_name: Internal benchmark name (e.g., 'rocfft')
            display_name: Display name for reports (e.g., 'ROCfft'), defaults to benchmark_name
        """
        self.benchmark_name = benchmark_name
        self.display_name = display_name or benchmark_name.upper()

        # Environment variables
        self.therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
        self.artifact_run_id = os.getenv("ARTIFACT_RUN_ID")
        self.amdgpu_families = os.getenv("AMDGPU_FAMILIES")
        self.script_dir = Path(__file__).resolve().parent
        self.therock_dir = self.script_dir.parent.parent.parent.parent

        # Initialize test client (will be set in run())
        self.client = None

    def execute_command(
        self,
        cmd: List[str],
        log_file_handle: IO,
        env: Dict[str, str] = None,
        cwd: Path = None,
    ) -> None:
        """Execute a command and stream output to log file.

        Args:
            cmd: Command list to execute
            log_file_handle: File handle to write output
            env: Optional environment variables to set
            cwd: Optional working directory (defaults to self.therock_dir)

        Raises:
            TestExecutionError: If command fails with non-zero exit code
        """
        working_dir = cwd if cwd is not None else self.therock_dir
        log.info(f"++ Exec [{working_dir}]$ {shlex.join(cmd)}")
        log_file_handle.write(f"{shlex.join(cmd)}\n")

        # Merge custom env with current environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        for line in process.stdout:
            log.info(line.strip())
            log_file_handle.write(f"{line}")

        process.wait()

        if process.returncode != 0:
            raise TestExecutionError(
                f"Command failed with exit code {process.returncode}\n"
                f"Command: {shlex.join(cmd)}\n"
                f"Working directory: {working_dir}\n"
                f"Check log file for details"
            )

    def _detect_gpu_count(self) -> int:
        """Detect the number of available GPUs using HardwareDetector.

        Returns:
            Number of GPUs detected

        Raises:
            RuntimeError: If no GPUs detected or detection fails
        """
        try:
            detector = HardwareDetector()
            gpu_list = detector.detect_gpu()
            gpu_count = len(gpu_list)

            if gpu_count == 0:
                raise RuntimeError(
                    "No GPUs detected. Benchmarks require at least one GPU. "
                    "Ensure ROCm drivers are installed and GPU devices are accessible."
                )

            log.info(f"Detected {gpu_count} GPU(s)")
            return gpu_count

        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to detect GPUs: {e}. "
                "Ensure ROCm drivers are installed and GPU devices are accessible."
            ) from e

    def _validate_openmpi(self) -> None:
        """Check if OpenMPI is installed and available in the system.

        Raises:
            TestExecutionError: If OpenMPI (mpirun) is not found
        """
        if not shutil.which("mpirun"):
            raise TestExecutionError(
                "OpenMPI not found in system\n"
                "Ensure OpenMPI is installed and 'mpirun' is available in PATH"
            )
        log.info("OpenMPI validated: mpirun found in system")

    def create_test_result(
        self,
        test_name: str,
        subtest_name: str,
        status: str,
        score: float,
        unit: str,
        flag: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a standardized test result dictionary.

        Args:
            test_name: Benchmark name
            subtest_name: Specific test identifier
            status: Test status ('PASS' or 'FAIL')
            score: Performance metric value
            unit: Unit of measurement (e.g., 'ms', 'GFLOPS', 'GB/s')
            flag: 'H' (higher is better) or 'L' (lower is better)
            **kwargs: Additional test-specific parameters (batch_size, ngpu, mode, etc.)

        Returns:
            Dict[str, Any]: Test result dictionary with test data and configuration
        """
        # Extract common parameters with defaults
        batch_size = kwargs.get("batch_size", 0)
        ngpu = kwargs.get("ngpu", 1)

        # Build test config with all parameters
        test_config = {
            "test_name": test_name,
            "sub_test_name": subtest_name,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "environment_dependencies": [],
            "batch_size": batch_size,
            "ngpu": ngpu,
        }

        # Add any additional kwargs to test_config
        for key, value in kwargs.items():
            if key not in ["batch_size", "ngpu"]:
                test_config[key] = value

        return {
            "test_name": test_name,
            "subtest": subtest_name,
            "batch_size": batch_size,
            "ngpu": ngpu,
            "status": status,
            "score": float(score),
            "unit": unit,
            "flag": flag,
            "test_config": test_config,
        }

    def calculate_statistics(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate test statistics from results.

        Args:
            test_results: List of test result dictionaries with 'status' key

        Returns:
            Dictionary with:
                - passed: Number of passed tests
                - failed: Number of failed tests
                - total: Total number of tests
                - overall_status: 'PASS' if no failures, else 'FAIL'
        """
        passed = sum(1 for r in test_results if r.get("status") == "PASS")
        failed = sum(1 for r in test_results if r.get("status") == "FAIL")
        overall_status = "PASS" if failed == 0 else "FAIL"

        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_results),
            "overall_status": overall_status,
        }

    def upload_results(
        self, test_results: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> bool:
        """Upload results to API and save locally."""
        log.info("Uploading Results to API")
        success = self.client.upload_results(
            test_name=f"{self.benchmark_name}_benchmark",
            test_results=test_results,
            test_status=stats["overall_status"],
            test_metadata={
                "artifact_run_id": self.artifact_run_id,
                "amdgpu_families": self.amdgpu_families,
                "benchmark_name": self.benchmark_name,
                "total_subtests": stats["total"],
                "passed_subtests": stats["passed"],
                "failed_subtests": stats["failed"],
            },
            save_local=True,
            output_dir=str(self.script_dir / "results"),
        )

        if success:
            log.info("Results uploaded successfully")
        else:
            log.info("Results saved locally only (API upload disabled or failed)")

        return success

    def compare_with_lkg(self, tables: Any) -> Any:
        """Compare results with Last Known Good baseline."""
        log.info("Comparing results with LKG")

        if isinstance(tables, list):
            # Compare each table with LKG
            final_tables = []
            for table in tables:
                if table._rows:
                    final_table = self.client.compare_results(
                        test_name=self.benchmark_name, table=table
                    )
                    log.info(f"\n{final_table}")
                    final_tables.append(final_table)
                else:
                    log.warning(f"Table '{table.title}' has no results, skipping")
            return final_tables

        # Single table
        final_table = self.client.compare_results(
            test_name=self.benchmark_name, table=tables
        )
        log.info(f"\n{final_table}")
        return final_table

    def write_step_summary(
        self, final_tables: Any, status_info: Dict[str, Any]
    ) -> None:
        """Write results to GitHub Actions step summary.

        Args:
            final_tables: Results table(s) with LKG comparison
            status_info: Dictionary from determine_final_status()
        """
        summary = (
            f"### {self.display_name} Benchmark Results\n\n"
            f"**Status:** {status_info['final_status']} | "
            f"**Passed:** {status_info['pass_count']}/{status_info['total_count']} | "
            f"**Failed:** {status_info['fail_count']}/{status_info['total_count']}"
        )

        if status_info["unknown_count"] > 0:
            summary += f" | **Unknown:** {status_info['unknown_count']}/{status_info['total_count']}"

        summary += "\n\n"

        if isinstance(final_tables, list):
            # Multiple tables - add each one
            for table in final_tables:
                summary += (
                    f"<details>\n"
                    f"<summary>{table.title}</summary>\n\n"
                    f"```\n{table}\n```\n\n"
                    f"</details>\n\n"
                )
        else:
            # Single table
            summary += (
                f"<details>\n"
                f"<summary>View detailed results ({status_info['total_count']} tests)</summary>\n\n"
                f"```\n{final_tables}\n```\n\n"
                f"</details>"
            )

        # Write to GitHub Actions step summary
        gha_append_step_summary(summary)

    def determine_final_status(self, final_tables: Any) -> Dict[str, Any]:
        """Determine final test status from results table(s).

        Returns:
            dict: {
                'final_status': str - Overall status ('PASS', 'FAIL', or 'UNKNOWN')
                'fail_count': int - Number of tests that failed LKG comparison
                'unknown_count': int - Number of tests with no baseline
                'pass_count': int - Number of tests that passed LKG comparison
                'total_count': int - Total number of tests
                'failed_tests': list - Names of tests that failed
                'unknown_tests': list - Names of tests with no baseline
            }
        """
        tables = final_tables if isinstance(final_tables, list) else [final_tables]

        fail_count = 0
        unknown_count = 0
        pass_count = 0
        failed_tests = []
        unknown_tests = []

        for table in tables:
            if "FinalResult" not in table.field_names:
                raise ValueError(f"Table '{table.title}' missing 'FinalResult' column")

            result_idx = table.field_names.index("FinalResult")
            name_idx = 0  # Assume first column is the test name/identifier

            # Extract results and test names
            results = [row[result_idx] for row in table._rows]
            test_names = [row[name_idx] for row in table._rows]

            # Count statuses
            fail_count += results.count("FAIL")
            unknown_count += results.count("UNKNOWN")
            pass_count += results.count("PASS")

            # Collect test names by status
            failed_tests.extend([test_names[i] for i, r in enumerate(results) if r == "FAIL"])
            unknown_tests.extend([test_names[i] for i, r in enumerate(results) if r == "UNKNOWN"])

        if unknown_count > 0 and fail_count == 0:
            log.warning("Some results have UNKNOWN status (no LKG data available)")

        final_status = (
            "FAIL" if fail_count > 0 else ("UNKNOWN" if unknown_count > 0 else "PASS")
        )

        return {
            "final_status": final_status,
            "fail_count": fail_count,
            "unknown_count": unknown_count,
            "pass_count": pass_count,
            "total_count": fail_count + unknown_count + pass_count,
            "failed_tests": failed_tests,
            "unknown_tests": unknown_tests,
        }

    def run(self) -> None:
        """Execute benchmark workflow.

        Raises:
            TestExecutionError: If benchmark execution encounters errors (missing files, etc.)
            TestResultError: If benchmarks run successfully but results show failures

        Note:
            On success, returns normally (exit code 0)
            On failure, raises exception (exit code 1)
        """
        log.info(f"Initializing {self.display_name} Benchmark Test")

        # Initialize benchmark client and print system info
        self.client = BenchmarkClient(auto_detect=True)
        self.client.print_system_summary()

        # Run benchmarks (implemented by child class)
        self.run_benchmarks()

        # Parse results (implemented by child class)
        test_results, tables = self.parse_results()

        # Validate test results structure
        if not test_results:
            raise TestResultError(
                "No test results found\n"
                "Ensure benchmarks were executed successfully and results were parsed"
            )

        # Calculate statistics
        stats = self.calculate_statistics(test_results)
        log.info(f"Test Summary: {stats['passed']} passed, {stats['failed']} failed")

        # Upload results
        self.upload_results(test_results, stats)

        # Compare with LKG (compares each table individually and prints results)
        final_tables = self.compare_with_lkg(tables)

        # Determine final status (do this BEFORE writing summary so we have correct counts)
        status_info = self.determine_final_status(final_tables)
        log.info(
            f"Final Status: {status_info['final_status']} "
            f"(PASS: {status_info['pass_count']}, "
            f"FAIL: {status_info['fail_count']}, "
            f"UNKNOWN: {status_info['unknown_count']})"
        )
        sys.stdout.flush()  # Ensure final status is displayed

        # Write results to GitHub Actions step summary
        self.write_step_summary(final_tables, status_info)

        # Flush output streams to ensure proper display ordering
        sys.stdout.flush()
        sys.stderr.flush()

        # Raise exception if benchmarks failed
        if status_info["final_status"] != "PASS":
            if status_info["fail_count"] > 0 and status_info["unknown_count"] > 0:
                failed_list = ", ".join(status_info["failed_tests"])
                unknown_list = ", ".join(status_info["unknown_tests"])
                raise TestResultError(
                    f"Benchmark test failed: {status_info['fail_count']} FAIL, "
                    f"{status_info['unknown_count']} UNKNOWN out of {status_info['total_count']} tests\n"
                    f"Failed tests: {failed_list}\n"
                    f"Unknown tests: {unknown_list}\n"
                    f"Performance regressions detected (FAIL) and missing baselines (UNKNOWN)"
                )
            elif status_info["fail_count"] > 0:
                failed_list = ", ".join(status_info["failed_tests"])
                raise TestResultError(
                    f"Benchmark test failed: {status_info['fail_count']} out of {status_info['total_count']} tests failed\n"
                    f"Failed tests: {failed_list}\n"
                    f"Performance regressions detected"
                )
            else:  # unknown_count > 0
                unknown_list = ", ".join(status_info["unknown_tests"])
                raise TestResultError(
                    f"Benchmark test status unknown: {status_info['unknown_count']} out of {status_info['total_count']} tests have no baseline\n"
                    f"Unknown tests: {unknown_list}\n"
                    f"No baseline data available for comparison (expected for new benchmarks)"
                )


def run_benchmark_main(benchmark_instance):
    """Run benchmark with standard error handling.

    Args:
        benchmark_instance: Instance of a benchmark test class

    Raises:
        TestExecutionError: If benchmark execution fails
        TestResultError: If benchmark results show failures
    """
    benchmark_instance.run()
