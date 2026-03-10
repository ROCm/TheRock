# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unified Communication X (UCX) ROCm integration tests.

Runs the pre-built UCX gtest binary to validate ROCm integration.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class UcxTest(FunctionalBase):
    """UCX ROCm integration tests."""

    GTEST_FILTER = "*rocm*"

    def __init__(self):
        super().__init__(test_name="ucx", display_name="UCX Test")

        self.results_json = self.script_dir / "ucx_results.json"

        # UCX build directory (populated by TheRock build system)
        self.ucx_build_dir = (
            self.therock_dir / "external-builds" / "ucx" / "ucx" / "build"
        )

    def run_tests(self) -> None:
        """Run UCX gtest using the pre-built binary, save results to JSON."""
        log.info(f"Running {self.display_name}")

        # Locate pre-built UCX gtest binary
        gtest_path = self.ucx_build_dir / "test" / "gtest" / "gtest"
        if not gtest_path.exists():
            raise TestExecutionError(
                f"UCX gtest binary not found at {gtest_path}\n"
                "Ensure TheRock was built with UCX gtest enabled"
            )
        log.info(f"Using UCX gtest binary: {gtest_path}")

        cmd = [
            str(gtest_path),
            f"--gtest_filter={self.GTEST_FILTER}",
            f"--gtest_output=json:{self.results_json}",
        ]

        # Set LD_LIBRARY_PATH to find ROCm libraries
        env = self.get_rocm_env()

        return_code, output = self.execute_command(cmd, cwd=self.ucx_build_dir, env=env)

        if return_code != 0:
            raise TestExecutionError(
                f"UCX gtest execution failed with return code {return_code}\n"
                f"Check logs for details"
            )

        log.info(f"{self.display_name} execution complete")

    def parse_results(self) -> List[Dict[str, Any]]:
        """Parse gtest JSON output and return results.

        Returns:
            List of test result dictionaries
        """
        log.info(f"Parsing {self.display_name} Results")

        try:
            with open(self.results_json, "r") as f:
                gtest_data = json.load(f)
        except FileNotFoundError:
            raise TestExecutionError(
                f"Gtest results file not found: {self.results_json}\n"
                f"Ensure tests were executed successfully"
            )
        except json.JSONDecodeError as e:
            raise TestExecutionError(
                f"Failed to parse gtest JSON: {e}\n"
                f"Check if gtest completed successfully"
            )

        test_results = []

        # Parse gtest JSON structure
        for testsuite in gtest_data.get("testsuites", []):
            suite_name = testsuite.get("name", "unknown_suite")

            for testcase in testsuite.get("testsuite", []):
                case_name = testcase.get("name", "unknown_case")
                full_name = f"{suite_name}.{case_name}"

                # Determine status from gtest result
                if testcase.get("failures"):
                    status = "FAIL"
                elif testcase.get("skipped"):
                    status = "SKIP"
                else:
                    status = "PASS"

                test_results.append(
                    self.create_test_result(
                        test_name=self.test_name,
                        subtest_name=full_name,
                        status=status,
                        suite=suite_name,
                    )
                )

        log.info(f"Parsed {len(test_results)} test results")
        return test_results


if __name__ == "__main__":
    run_functional_main(UcxTest())
