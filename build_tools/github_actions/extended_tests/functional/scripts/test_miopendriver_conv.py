"""
MIOpen Driver Convolution Functional Test

Tests MIOpenDriver convolution operations (Forward and Backward) to ensure
correct functionality across different GPU architectures.
"""

import json
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

        self.results_json = self.script_dir / "miopendriver_conv_results.json"

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
        """Run MIOpen driver convolution tests and save results to JSON."""
        log.info(f"Running {self.display_name} Tests")

        # Detect GPU architecture using HardwareDetector
        # Returns first discrete GPU. Use ROCR_VISIBLE_DEVICES to control which GPU if needed.
        detector = HardwareDetector()
        gfx_id = detector.get_gpu_architecture()
        log.info(f"Detected GPU: {gfx_id}")

        miopen_driver = Path(self.therock_bin_dir) / "MIOpenDriver"
        if not miopen_driver.exists():
            raise TestExecutionError(
                f"MIOpenDriver not found at {miopen_driver}\n"
                f"Ensure MIOpen is installed correctly"
            )

        # Calculate total number of tests for progress indicator
        total_tests = sum(len(self.tests_cmd[suite]) for suite in self.tests_list)
        current_test = 0
        log.info(f"Total {self.display_name} tests to run: {total_tests}")

        # Store results as we execute
        all_results = []

        for test_suite in self.tests_list:
            log.info(f"Running test suite: {test_suite}")

            for i, cmd_str in enumerate(self.tests_cmd[test_suite], 1):
                current_test += 1

                # Build full command with MIOpenDriver path
                full_cmd = f"{miopen_driver} {cmd_str}"

                # Add GPU-specific flags if needed
                if "Backward_Conv" in test_suite and gfx_id in self.gpu_specific_flags:
                    backward_flags = self.gpu_specific_flags[gfx_id].get(
                        "backward_flags", ""
                    )
                    full_cmd = f"{full_cmd} {backward_flags}"

                cmd = shlex.split(full_cmd)

                # Progress indicator
                test_case_name = f"{test_suite}_case{i}"
                log.info(f"[{current_test}/{total_tests}] Running {test_case_name}")
                log.info(f"++ Exec [{self.therock_dir}]$ {shlex.join(cmd)}")

                return_code = None
                error_message = None

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

                    process.wait()
                    return_code = process.returncode

                except Exception as e:
                    log.error(f"Error running command: {e}")
                    error_message = str(e)
                    return_code = -1

                # Determine status based on return code
                status = "PASS" if return_code == 0 else "FAIL"

                # Store result immediately
                result = {
                    "test_suite": test_suite,
                    "test_case": test_case_name,
                    "command": cmd_str,
                    "command_index": i,
                    "return_code": return_code,
                    "status": status,
                }
                if error_message:
                    result["error"] = error_message

                all_results.append(result)
                log.info(f"[{current_test}/{total_tests}] {test_case_name}: {status}")

        # Write all results to JSON file
        with open(self.results_json, "w") as f:
            json.dump(all_results, f, indent=2)

        log.info(f"{self.display_name} results saved to {self.results_json}")
        log.info(f"{self.display_name} test execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable, int]:
        """Parse test results from JSON file.

        Returns:
            tuple: (test_results list, detailed PrettyTable, number of test suites)
        """
        log.info(f"Parsing {self.display_name} Results")

        # Setup detailed table - show each individual test case
        detailed_table = PrettyTable()
        detailed_table.field_names = ["TestSuite", "TestCase", "Status"]

        test_results = []

        try:
            # Read results from JSON file
            with open(self.results_json, "r") as f:
                json_results = json.load(f)

            if not isinstance(json_results, list):
                raise TestExecutionError(
                    "Results JSON is not a list\n" "Check results file format"
                )

            # Process each result with safe key access
            for idx, result in enumerate(json_results):
                if not isinstance(result, dict):
                    log.warning(f"Result {idx} is not a dictionary, skipping")
                    continue

                # Use .get() with defaults to handle missing keys gracefully
                test_suite = result.get("test_suite", "unknown_suite")
                test_case = result.get("test_case", f"unknown_case_{idx}")
                status = result.get("status", "FAIL")  # Default to FAIL if unknown
                command = result.get("command", "")
                command_index = result.get("command_index", idx + 1)

                # Log warning if critical keys are missing
                if "test_suite" not in result or "test_case" not in result:
                    log.warning(
                        f"Result {idx} missing critical keys (test_suite/test_case)"
                    )

                # Add to detailed table
                detailed_table.add_row([test_suite, test_case, status])

                # Add to results list using helper
                test_results.append(
                    self.create_test_result(
                        test_name=self.test_name,
                        subtest_name=test_case,
                        status=status,
                        suite=test_suite,
                        command_index=command_index,
                        command=command,
                    )
                )

        except FileNotFoundError:
            raise TestExecutionError(
                f"Results JSON file not found: {self.results_json}\n"
                f"Ensure tests were executed successfully"
            )
        except json.JSONDecodeError as e:
            raise TestExecutionError(
                f"Error parsing results JSON: {e}\n"
                f"Check if results file is valid JSON"
            )
        except OSError as e:
            raise TestExecutionError(
                f"Error reading results file: {e}\n"
                f"Check file permissions and disk space"
            )

        if not test_results:
            raise TestExecutionError(
                "No valid test results found in JSON file\n"
                "Check if tests executed successfully and results were saved"
            )

        num_suites = len(self.tests_list)
        return test_results, detailed_table, num_suites


if __name__ == "__main__":
    run_functional_test_main(MIOpenDriverConvTest())
