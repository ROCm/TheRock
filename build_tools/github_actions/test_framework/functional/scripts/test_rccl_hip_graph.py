"""
RCCL HIP Graph Functional Test

Tests RCCL HIP Graph functionality by building and running RCCL test executables.
"""

import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_test_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class RCCLHIPGraphTest(FunctionalBase):
    """RCCL HIP Graph functional test."""

    def __init__(self):
        super().__init__(test_name="rccl_hip_graph", display_name="RCCL HIP Graph")

        self.log_file = self.script_dir / "rccl_hip_graph.log"

        # Load test configuration from JSON
        config = self.load_config("rccl_hip_graph.json")

        # Parse configuration
        self.repo_url = config.get("repository_url")
        self.repo_branch = config.get("repository_branch")
        self.repo_name = config.get("repository_name")
        self.test_args = config.get("test_args", "")
        self.clone_dir = Path.home() / self.repo_name

    def build(self) -> None:
        """Build the RCCL tests.

        This method is automatically called by the base class before run_tests().
        It clones the repository and builds the tests.
        """
        log.info("=" * 80)
        log.info("Building Tests")
        log.info("=" * 80)

        # Clone repository
        success, message = self.clone_repository(
            self.repo_url,
            self.clone_dir,
            branch=self.repo_branch,
        )
        if not success:
            raise TestExecutionError(
                f"Failed to clone repository: {message}",
                action="Check network connectivity and repository URL",
            )

        # Build RCCL tests
        build_cmd = ["make", "-j"]
        log.info(f"Building with: {shlex.join(build_cmd)}")

        # Create a temporary build log file
        build_log = self.script_dir / "rccl_build.log"
        with open(build_log, "w") as f:
            return_code = self.execute_command(build_cmd, self.clone_dir, f)

        if return_code != 0:
            raise TestExecutionError(
                f"Build failed with return code {return_code}",
                action=f"Check build errors in {build_log}",
            )

        log.info("RCCL HIP Graph build completed successfully")

    def run_tests(self) -> None:
        """Run RCCL HIP Graph tests and save output to log file."""
        log.info(f"Running {self.display_name} Tests")

        # Verify build was successful
        build_dir = self.clone_dir / "build"
        if not build_dir.exists() or not any(build_dir.iterdir()):
            raise TestExecutionError(
                f"Build directory empty or not found: {build_dir}",
                action="Check if build succeeded",
            )

        # Get all executables in build directory
        executables = [
            f for f in build_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)
        ]

        if not executables:
            raise TestExecutionError(
                f"No executable files found in {build_dir}",
                action="Check if build produced executables",
            )

        with open(self.log_file, "w") as f:
            f.write(f"{'='*80}\n")
            f.write(f"Running Test Executables\n")
            f.write(f"{'='*80}\n\n")

            for exe in executables:
                log.info(f"Running test executable: {exe.name}")
                f.write(f"\n{'-'*80}\n")
                f.write(f"Test Executable: {exe.name}\n")
                f.write(f"{'-'*80}\n\n")

                # Build command with arguments
                test_cmd = [str(exe)] + shlex.split(self.test_args)
                self.execute_command(test_cmd, build_dir, f)

        log.info("RCCL HIP Graph test execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable, int]:
        """Parse test results from log file.

        Returns:
            tuple: (test_results list, detailed PrettyTable, number of test suites)
        """
        log.info("Parsing Results")

        # Setup detailed table
        detailed_table = PrettyTable()
        detailed_table.field_names = ["TestSuite", "TestCase", "Status"]

        test_results = []

        # Pass/fail criteria
        pass_patterns = [r"\bOK\b"]
        fail_patterns = [r"\baborted\b"]

        try:
            with open(self.log_file, "r") as f:
                content = f.read()

            # Find all test executable sections
            test_sections = re.split(r"Test Executable: (.+?)\n", content)

            # Process each test executable
            for i in range(1, len(test_sections), 2):
                if i + 1 >= len(test_sections):
                    break

                test_name = test_sections[i].strip()
                test_content = test_sections[i + 1]

                # Extract the section until next test or end
                next_test = re.search(r"Test Executable:", test_content)
                if next_test:
                    test_content = test_content[: next_test.start()]

                # Determine status
                status = "FAIL"  # Default to FAIL

                # Check for pass patterns
                if any(
                    re.search(pattern, test_content, re.IGNORECASE)
                    for pattern in pass_patterns
                ):
                    status = "PASS"

                # Check for fail patterns (overrides pass)
                if any(
                    re.search(pattern, test_content, re.IGNORECASE)
                    for pattern in fail_patterns
                ):
                    status = "FAIL"

                # Check return code
                return_code_match = re.search(r"Return code:\s*(\d+)", test_content)
                if return_code_match:
                    return_code = int(return_code_match.group(1))
                    if return_code != 0:
                        status = "FAIL"

                # Add to detailed table
                detailed_table.add_row(["rccl_hip_graph", test_name, status])

                # Add to results list
                test_results.append(
                    self.create_test_result(
                        test_name=self.test_name,
                        subtest_name=test_name,
                        status=status,
                        suite="rccl_hip_graph",
                    )
                )

        except FileNotFoundError:
            raise TestExecutionError(
                f"Log file not found: {self.log_file}",
                action="Ensure tests were executed successfully",
            )
        except OSError as e:
            raise TestExecutionError(
                f"Error reading log file: {e}",
                action="Check file permissions and disk space",
            )

        num_suites = 1
        return test_results, detailed_table, num_suites


if __name__ == "__main__":
    run_functional_test_main(RCCLHIPGraphTest())
