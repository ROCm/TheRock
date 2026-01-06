"""
MIOpen Driver Convolution Functional Test

Tests MIOpenDriver convolution operations (Forward and Backward) to ensure
correct functionality across different GPU architectures.
"""

import json
import os
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
from utils.exceptions import TestExecutionError


class MIOpenDriverConvTest(FunctionalBase):
    """MIOpen Driver convolution functional test."""

    def __init__(self):
        super().__init__(
            test_name="miopen_driver_conv",
            display_name="MIOpen Driver Convolution"
        )
        
        self.log_file = self.script_dir / "miopendriver_conv.log"

        # Load test configurations from JSON
        config = self.load_config("miopen_driver_conv.json")

        # Parse test suites
        test_suites = config.get("test_suites", {})
        self.tests_cmd = {}
        self.envs = {}
        self.tests_list = []

        for suite_name, suite_config in test_suites.items():
            self.tests_list.append(suite_name)
            self.tests_cmd[suite_name] = suite_config.get("commands", [])
            self.envs[suite_name] = suite_config.get("algorithm", "")

        # Load GPU-specific flags
        self.gpu_specific_flags = config.get("gpu_specific_flags", {})

    def get_gpu_id(self) -> str:
        """Detect GPU ID using rocminfo."""
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract GPU name (e.g., gfx906, gfx90a, gfx942)
            match = re.search(r'Name:\s+(gfx\w+)', result.stdout)
            if match:
                return match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("Could not detect GPU ID, assuming default")
        
        return "unknown"

    def run_tests(self) -> None:
        """Run MIOpen driver convolution tests and save output to log file."""
        log.info(f"Running {self.display_name} Tests")

        gpu_id = self.get_gpu_id()
        log.info(f"Detected GPU: {gpu_id}")

        miopen_driver = f"{self.therock_bin_dir}/MIOpenDriver"
        if not Path(miopen_driver).exists():
            raise TestExecutionError(
                f"MIOpenDriver not found at {miopen_driver}",
                action="Ensure MIOpen is installed correctly"
            )

        with open(self.log_file, "w+") as f:
            for test_suite in self.tests_list:
                log.info(f"Running test suite: {test_suite}")
                f.write(f"\n{'='*80}\n")
                f.write(f"Test Suite: {test_suite}\n")
                f.write(f"{'='*80}\n\n")

                # Set environment variable for specific algorithm
                env = os.environ.copy()
                env['MIOPEN_FIND_ENFORCE'] = self.envs[test_suite]

                for cmd_str in self.tests_cmd[test_suite]:
                    # Build full command with MIOpenDriver path
                    full_cmd = f"{miopen_driver} {cmd_str}"
                    
                    # Add GPU-specific flags if needed
                    if 'Backward_Conv' in test_suite and gpu_id in self.gpu_specific_flags:
                        backward_flags = self.gpu_specific_flags[gpu_id].get("backward_flags", "")
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
                            env=env
                        )

                        for line in process.stdout:
                            log.info(line.strip())
                            f.write(line)

                        process.wait()
                        f.write(f"\nReturn code: {process.returncode}\n\n")

                    except Exception as e:
                        log.error(f"Error running command: {e}")
                        f.write(f"ERROR: {e}\n\n")

        log.info("Test execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse test results from log file.

        Returns:
            tuple: (test_results list, PrettyTable object)
        """
        log.info("Parsing Results")

        # Setup table
        field_names = [
            "TestSuite",
            "CommandIndex",
            "Status",
            "PassCount",
            "FailCount",
            "TotalCommands"
        ]
        table = PrettyTable(field_names)

        test_results = []

        try:
            with open(self.log_file, "r") as f:
                content = f.read()

            # Parse results for each test suite
            for test_suite in self.tests_list:
                # Find the section for this test suite
                pattern = re.compile(
                    rf"Test Suite: {test_suite}.*?(?=Test Suite:|$)",
                    re.DOTALL
                )
                match = pattern.search(content)
                
                if not match:
                    log.warning(f"Could not find results for {test_suite}")
                    continue

                suite_content = match.group(0)
                
                # Count commands and their results
                total_commands = len(self.tests_cmd[test_suite])
                pass_count = 0
                fail_count = 0

                # Look for "PASSED" or return code 0 indicators
                # MIOpenDriver typically outputs success indicators
                for i, cmd_str in enumerate(self.tests_cmd[test_suite], 1):
                    # Build the command pattern to search for
                    miopen_driver = f"{self.therock_bin_dir}/MIOpenDriver"
                    full_cmd = f"{miopen_driver} {cmd_str}"
                    
                    # Simple heuristic: if command appears and no error follows, assume pass
                    # This is a simplified parser - adjust based on actual MIOpenDriver output
                    if f"Command: {full_cmd}" in suite_content or cmd_str in suite_content:
                        # Check for return code 0 or success indicators
                        if "Return code: 0" in suite_content or "PASSED" in suite_content:
                            pass_count += 1
                            status = "PASS"
                        else:
                            fail_count += 1
                            status = "FAIL"
                    else:
                        fail_count += 1
                        status = "FAIL"

                # Determine overall suite status
                overall_status = "PASS" if fail_count == 0 else "FAIL"

                # Add to table
                table.add_row([
                    test_suite,
                    f"1-{total_commands}",
                    overall_status,
                    pass_count,
                    fail_count,
                    total_commands
                ])

                # Add to results
                test_results.append({
                    "test_name": self.test_name,
                    "subtest_name": test_suite,
                    "status": overall_status,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "total_commands": total_commands
                })

        except FileNotFoundError:
            raise TestExecutionError(
                f"Log file not found: {self.log_file}",
                action="Ensure tests were executed successfully"
            )
        except OSError as e:
            raise TestExecutionError(
                f"Error reading log file: {e}",
                action="Check file permissions and disk space"
            )

        return test_results, table


if __name__ == "__main__":
    run_functional_test_main(MIOpenDriverConvTest())
