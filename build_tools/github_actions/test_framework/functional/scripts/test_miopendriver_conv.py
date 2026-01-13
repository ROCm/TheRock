"""
MIOpen Driver Convolution Functional Test

Tests MIOpenDriver convolution operations (Forward and Backward) to ensure
correct functionality across different GPU architectures.
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_test_main
from utils.logger import log
from utils.system.hardware import HardwareDetector
from utils.exceptions import TestExecutionError


class MIOpenDriverConvTest(FunctionalBase):
    """MIOpen Driver convolution functional test."""

    def __init__(self):
        super().__init__(
            test_name="miopen_driver_conv", display_name="MIOpen Driver Convolution"
        )

        self.log_file = self.script_dir / "miopendriver_conv.log"

        # Load test configurations from JSON
        config = self.load_config("miopen_driver_conv.json")

        # Parse test suites
        test_suites = config.get("test_suites", {})
        self.tests_cmd = {}
        self.tests_list = []

        for suite_name, suite_config in test_suites.items():
            self.tests_list.append(suite_name)
            self.tests_cmd[suite_name] = suite_config.get("commands", [])

        # Load GPU-specific flags
        self.gpu_specific_flags = config.get("gpu_specific_flags", {})

    def run_tests(self) -> None:
        """Run MIOpen driver convolution tests and save output to log file."""
        log.info(f"Running {self.display_name} Tests")

        # Detect GPU architecture using HardwareDetector
        detector = HardwareDetector()
        gfx_id = detector.get_gpu_architecture()
        log.info(f"Detected GPU: {gfx_id}")

        miopen_driver = f"{self.therock_bin_dir}/MIOpenDriver"
        if not Path(miopen_driver).exists():
            raise TestExecutionError(
                f"MIOpenDriver not found at {miopen_driver}",
                action="Ensure MIOpen is installed correctly",
            )

        with open(self.log_file, "w+") as f:
            for test_suite in self.tests_list:
                log.info(f"Running test suite: {test_suite}")
                f.write(f"\n{'='*80}\n")
                f.write(f"Test Suite: {test_suite}\n")
                f.write(f"{'='*80}\n\n")

                for cmd_str in self.tests_cmd[test_suite]:
                    # Build full command with MIOpenDriver path
                    full_cmd = f"{miopen_driver} {cmd_str}"

                    # Add GPU-specific flags if needed
                    if (
                        "Backward_Conv" in test_suite
                        and gfx_id in self.gpu_specific_flags
                    ):
                        backward_flags = self.gpu_specific_flags[gfx_id].get(
                            "backward_flags", ""
                        )
                        if backward_flags:
                            full_cmd = f"{full_cmd} {backward_flags}"

                    cmd = shlex.split(full_cmd)

                    log.info(f"++ Exec [{self.therock_dir}]$ {shlex.join(cmd)}")
                    f.write(f"\nCommand: {shlex.join(cmd)}\n")
                    f.write(f"-" * 80 + "\n")

                    try:
                        process = subprocess.Popen(
                            cmd,
                            cwd=self.therock_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )

                        for line in process.stdout:
                            log.info(line.strip())
                            f.write(line)

                        process.wait()
                        f.write(f"\nReturn code: {process.returncode}\n\n")

                    except Exception as e:
                        log.error(f"Error running command: {e}")
                        f.write(f"ERROR: {e}\n\n")

        log.info("MIOpenDriver convolution test execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable, int]:
        """Parse test results from log file.

        Returns:
            tuple: (test_results list, detailed PrettyTable, number of test suites)
        """
        log.info("Parsing Results")

        # Setup detailed table - show each individual test case
        detailed_table = PrettyTable()
        detailed_table.field_names = ["TestSuite", "TestCase", "Status"]

        test_results = []

        try:
            with open(self.log_file, "r") as f:
                content = f.read()

            # Parse results for each test suite
            for test_suite in self.tests_list:
                # Find the section for this test suite
                pattern = re.compile(
                    rf"Test Suite: {test_suite}.*?(?=Test Suite:|$)", re.DOTALL
                )
                match = pattern.search(content)

                if not match:
                    log.warning(f"Could not find results for {test_suite}")
                    continue

                suite_content = match.group(0)

                # Parse each command in the suite
                for i, cmd_str in enumerate(self.tests_cmd[test_suite], 1):
                    # Build the command pattern to search for
                    miopen_driver = f"{self.therock_bin_dir}/MIOpenDriver"
                    full_cmd = f"{miopen_driver} {cmd_str}"
                    test_case_name = f"{test_suite}_case{i}"

                    # Find this specific command in the suite content
                    cmd_pattern = re.escape(f"Command: {full_cmd}")
                    cmd_match = re.search(cmd_pattern, suite_content)

                    if cmd_match:
                        # Extract content after this command until next command or end
                        start_pos = cmd_match.end()
                        next_cmd_match = re.search(
                            r"\nCommand:", suite_content[start_pos:]
                        )
                        if next_cmd_match:
                            cmd_section = suite_content[
                                start_pos : start_pos + next_cmd_match.start()
                            ]
                        else:
                            cmd_section = suite_content[start_pos:]

                        # Check return code in THIS command's section only
                        return_code_match = re.search(
                            r"Return code:\s*(\d+)", cmd_section
                        )
                        if return_code_match:
                            return_code = int(return_code_match.group(1))
                            status = "PASS" if return_code == 0 else "FAIL"
                        elif "PASSED" in cmd_section:
                            status = "PASS"
                        elif "FAILED" in cmd_section or "ERROR" in cmd_section:
                            status = "FAIL"
                        else:
                            status = "FAIL"  # Unknown result, assume fail
                    else:
                        status = "FAIL"  # Command not found in log

                    # Add each individual test case to detailed table
                    detailed_table.add_row([test_suite, test_case_name, status])

                    # Add each test case to results list using helper
                    test_results.append(
                        self.create_test_result(
                            test_name=self.test_name,
                            subtest_name=test_case_name,
                            status=status,
                            suite=test_suite,
                            command_index=i,
                            command=cmd_str,
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

        num_suites = len(self.tests_list)
        return test_results, detailed_table, num_suites


if __name__ == "__main__":
    run_functional_test_main(MIOpenDriverConvTest())
