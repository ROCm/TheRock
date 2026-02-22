"""
Unified Communication X (UCX) ROCm integration tests

Clones, builds and runs UCX gtest, collects results, and uploads to results API.
"""

import json
import os
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

    def __init__(self):
        super().__init__(test_name="ucx", display_name="UCX Test")

        self.results_json = self.script_dir / "ucx_results.json"
        # Resolve to absolute path (required by configure --prefix)
        self.ucx_dir = self.rocm_path / "bin" / "ucx"
        self.ucx_build_dir = self.ucx_dir / "build"

        # Load test configuration from JSON
        config = self.load_config("ucx.json")
        self.git_url = config["git_url"]
        self.git_branch = config["git_branch"]
        self.gtest_filter = config["gtest_filter"]

    def _build_ucx(self) -> None:
        """Build UCX with ROCm support."""
        log.info("Building UCX with ROCm support")

        build_steps = [
            {
                "name": "autogen",
                "cmd": ["./autogen.sh"],
                "cwd": self.ucx_dir,
            },
            {
                "name": "mkdir build",
                "cmd": ["mkdir", "-p", "build"],
                "cwd": self.ucx_dir,
            },
            {
                "name": "configure",
                "cmd": [
                    "../contrib/configure-release",
                    "--disable-logging",
                    "--disable-debug",
                    "--disable-assertions",
                    "--enable-params-check",
                    f"--prefix={self.ucx_build_dir}",
                    "--without-knem",
                    "--without-cuda",
                    f"--with-rocm={self.rocm_path}",
                    "--enable-gtest",
                    "--without-gdrcopy",
                    "--without-java",
                ],
                "cwd": self.ucx_build_dir,
            },
            {
                "name": "make",
                "cmd": ["make", f"-j{os.cpu_count()}"],
                "cwd": self.ucx_build_dir,
            },
            {
                "name": "make install",
                "cmd": ["make", f"-j{os.cpu_count()}", "install"],
                "cwd": self.ucx_build_dir,
            },
        ]

        for step in build_steps:
            log.info(f"Running build step: {step['name']}")

            return_code = self.execute_command(step["cmd"], cwd=step["cwd"])

            if return_code != 0:
                raise TestExecutionError(f"UCX build failed at step '{step['name']}'")

        log.info("UCX build completed")

    def run_tests(self) -> None:
        """Clone, build and run UCX gtest, save results to JSON."""
        log.info(f"Running {self.display_name}")

        # Clone and build UCX
        self.clone_repository(
            git_url=self.git_url,
            branch=self.git_branch,
            target_dir=self.ucx_dir,
        )
        self._build_ucx()

        # Run gtest
        gtest_path = self.ucx_build_dir / "test" / "gtest" / "gtest"
        if not gtest_path.exists():
            raise TestExecutionError(
                f"UCX gtest not found at {gtest_path}\n"
                f"Ensure UCX build completed successfully"
            )

        cmd = [
            str(gtest_path),
            f"--gtest_filter={self.gtest_filter}",
            f"--gtest_output=json:{self.results_json}",
        ]

        # Set LD_LIBRARY_PATH to find ROCm libraries
        env = self.get_rocm_env()
        return_code = self.execute_command(cmd, cwd=self.ucx_build_dir, env=env)

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
