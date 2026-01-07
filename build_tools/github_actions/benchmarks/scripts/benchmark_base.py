"""Base class for benchmark tests with common functionality."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))  # benchmarks/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # github_actions/
from utils import BenchmarkClient
from utils.logger import log
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

        # Initialize test client (will be set in run())
        self.client = None

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

    def compare_with_lkg(self, table: PrettyTable) -> PrettyTable:
        """Compare results with Last Known Good baseline."""
        log.info("Comparing results with LKG")
        final_table = self.client.compare_results(
            test_name=self.benchmark_name, table=table
        )
        log.info(f"\n{final_table}")
        return final_table

    def write_step_summary(
        self, stats: Dict[str, Any], final_table: PrettyTable
    ) -> None:
        """Write results to GitHub Actions step summary."""
        gha_append_step_summary(
            f"## {self.display_name} Benchmark Results\n\n"
            f"**Status:** {stats['overall_status']} | "
            f"**Passed:** {stats['passed']}/{stats['total']} | "
            f"**Failed:** {stats['failed']}/{stats['total']}\n\n"
            f"<details>\n"
            f"<summary>View detailed results ({stats['total']} tests)</summary>\n\n"
            f"```\n{final_table}\n```\n\n"
            f"</details>"
        )

    def determine_final_status(self, final_table: PrettyTable) -> str:
        """Determine final test status from results table."""
        if "FinalResult" not in final_table.field_names:
            raise ValueError("The table does not have a 'FinalResult' column.")

        final_result_index = final_table.field_names.index("FinalResult")
        has_fail = any(row[final_result_index] == "FAIL" for row in final_table._rows)
        has_unknown = any(
            row[final_result_index] == "UNKNOWN" for row in final_table._rows
        )

        final_status = "FAIL" if has_fail else ("UNKNOWN" if has_unknown else "PASS")

        if has_unknown and not has_fail:
            log.warning(
                "Some results have UNKNOWN status (no LKG data available for comparison)"
            )

        return final_status

    def run(self) -> int:
        """Execute benchmark workflow and return exit code (0=PASS, 1=FAIL)."""
        log.info(f"Initializing {self.display_name} Benchmark Test")

        # Initialize benchmark client and print system info
        self.client = BenchmarkClient(auto_detect=True)
        self.client.print_system_summary()

        # Run benchmarks (implemented by child class)
        self.run_benchmarks()

        # Parse results (implemented by child class)
        test_results, table = self.parse_results()

        if not test_results:
            log.error("No test results found")
            return 1

        # Calculate statistics
        stats = self.calculate_statistics(test_results)
        log.info(f"Test Summary: {stats['passed']} passed, {stats['failed']} failed")

        # Upload results
        self.upload_results(test_results, stats)

        # Compare with LKG
        final_table = self.compare_with_lkg(table)

        # Write to GitHub Actions step summary
        self.write_step_summary(stats, final_table)

        # Determine final status
        final_status = self.determine_final_status(final_table)
        log.info(f"Final Status: {final_status}")

        # Return 0 only if PASS, otherwise return 1
        return 0 if final_status == "PASS" else 1


def run_benchmark_main(benchmark_instance):
    """Run benchmark with standard error handling.

    Raises:
        KeyboardInterrupt: If execution is interrupted by user
        Exception: If benchmark execution fails
    """
    try:
        exit_code = benchmark_instance.run()
        if exit_code != 0:
            raise RuntimeError(f"Benchmark failed with exit code {exit_code}")
    except KeyboardInterrupt:
        log.warning("\nExecution interrupted by user")
        raise
    except Exception as e:
        log.error(f"Execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise
