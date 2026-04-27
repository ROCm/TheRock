# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for benchmark tests with common functionality."""

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any
from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.logger import log
from utils.exceptions import TestExecutionError
from utils.extended_test_base import ExtendedTestBase, gha_append_step_summary


# TODO(lajagapp): Set to True once the results API network/firewall issue is
# resolved. (https://github.com/ROCm/TheRock/issues/3850)
ENABLE_RESULTS_API = False


class BenchmarkBase(ExtendedTestBase):
    """Base class providing common benchmark logic.

    Inherits shared infrastructure from ExtendedTestBase (execute_command,
    create_test_result, calculate_statistics, upload_results, etc.).

    Child classes must implement run_benchmarks() and parse_results().
    """

    def __init__(self, benchmark_name: str, display_name: str = None):
        """Initialize benchmark test.

        Args:
            benchmark_name: Internal benchmark name (e.g., 'rocfft')
            display_name: Display name for reports (e.g., 'ROCfft'), defaults to benchmark_name
        """
        super().__init__(benchmark_name, display_name or benchmark_name.upper())
        self.benchmark_name = benchmark_name
        self.script_dir = Path(__file__).resolve().parent

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
        """Create a standardized benchmark test result dictionary.

        Overrides ExtendedTestBase.create_test_result to enforce benchmark-specific
        required fields (score, unit, flag) and provide defaults for
        batch_size and ngpu.

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
        # Extract benchmark-specific parameters with defaults
        batch_size = kwargs.pop("batch_size", 0)
        ngpu = kwargs.pop("ngpu", 1)

        return super().create_test_result(
            test_name=test_name,
            subtest_name=subtest_name,
            status=status,
            score=float(score),
            unit=unit,
            flag=flag,
            batch_size=batch_size,
            ngpu=ngpu,
            **kwargs,
        )

    def upload_results(self, **kwargs) -> bool:
        """Upload results to API and save locally.

        Overrides ExtendedTestBase.upload_results to gate on ENABLE_RESULTS_API.
        """
        if not ENABLE_RESULTS_API:
            log.warning(
                "Results API is disabled temporarily (ENABLE_RESULTS_API=False). Skipping upload."
            )
            return False

        return super().upload_results(**kwargs)

    def compare_with_lkg(self, tables: Any) -> Any:
        """Compare results with Last Known Good baseline."""
        if not ENABLE_RESULTS_API:
            log.warning(
                "Results API is disabled temporarily (ENABLE_RESULTS_API=False). Skipping LKG comparison."
            )
            # Print raw tables so scores are still visible in the log.
            table_list = tables if isinstance(tables, list) else [tables]
            for table in table_list:
                if table._rows:
                    log.info(f"\n{table}")
            return None

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

    def write_step_summary(self, stats: Dict[str, Any], final_tables: Any) -> None:
        """Write results to GitHub Actions step summary."""
        summary = (
            f"## {self.display_name} Benchmark Results\n\n"
            f"**Status:** {stats['overall_status']} | "
            f"**Passed:** {stats['passed']}/{stats['total']} | "
            f"**Failed:** {stats['failed']}/{stats['total']}\n\n"
        )

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
                f"<summary>View detailed results ({stats['total']} tests)</summary>\n\n"
                f"```\n{final_tables}\n```\n\n"
                f"</details>"
            )

        gha_append_step_summary(summary)

    def determine_final_status(self, final_tables: Any) -> str:
        """Determine final test status from results table(s)."""
        tables = final_tables if isinstance(final_tables, list) else [final_tables]

        has_fail = has_unknown = False
        for table in tables:
            if "FinalResult" not in table.field_names:
                raise ValueError(f"Table '{table.title}' missing 'FinalResult' column")

            idx = table.field_names.index("FinalResult")
            results = [row[idx] for row in table._rows]
            has_fail = has_fail or "FAIL" in results
            has_unknown = has_unknown or "UNKNOWN" in results

        if has_unknown and not has_fail:
            log.warning("Some results have UNKNOWN status (no LKG data available)")

        return "FAIL" if has_fail else ("UNKNOWN" if has_unknown else "PASS")

    def run(self) -> int:
        """Execute benchmark workflow and return exit code (0=PASS, 1=FAIL)."""
        log.info(f"Initializing {self.display_name} Benchmark Test")

        # Run benchmarks (implemented by child class)
        self.run_benchmarks()

        # Parse results (implemented by child class)
        test_results, tables = self.parse_results()

        if not test_results:
            log.error("No test results found")
            return 1

        # Calculate statistics
        stats = self.calculate_statistics(test_results)
        log.info(f"Test Summary: {stats['passed']} passed, {stats['failed']} failed")

        # Upload results
        self.upload_results(
            test_results=test_results,
            stats=stats,
            test_type="benchmark",
            output_dir=str(self.script_dir / "results"),
            extra_metadata={
                "benchmark_name": self.benchmark_name,
                "total_subtests": stats["total"],
                "passed_subtests": stats["passed"],
                "failed_subtests": stats["failed"],
            },
        )

        # Compare with LKG (returns None when ENABLE_RESULTS_API is False)
        final_tables = self.compare_with_lkg(tables)

        if final_tables is not None:
            # Write to GitHub Actions step summary
            self.write_step_summary(stats, final_tables)
            # Determine final status from LKG comparison
            final_status = self.determine_final_status(final_tables)
        else:
            # API disabled — use pass/fail stats directly
            final_status = stats["overall_status"]

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
