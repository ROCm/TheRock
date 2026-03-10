#!/usr/bin/env python3
"""
ROCm Bandwidth Test (RBT) Test.

Executes the pre-built ROCm Bandwidth Test binary to validate GPU memory
bandwidth and peer-to-peer communication functionality.
"""

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class RBTTest(FunctionalBase):
    """ROCm Bandwidth Test functional test suite."""

    def __init__(self):
        super().__init__(test_name="rbt", display_name="RBT Test")

        self.results_json = self.script_dir / "rbt_results.json"
        self.log_file = self.script_dir / "rbt.log"

        # Load test configuration from JSON
        config = self.load_config("rbt.json")

        # Test configuration (env var overrides config)
        self.test_type = os.getenv("TEST_TYPE", config.get("test_type", "full"))
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"

        # Test definitions from config
        self.tests_by_type = config.get("tests", {})
        self.test_definitions = config.get("test_definitions", {})

    @staticmethod
    def _get_skip_reason(
        requires: Dict[str, Any], num_gpus: int, gfx_version: str
    ) -> str | None:
        """Return a skip-reason string if *requires* are NOT met, else None.

        Supported config keys: arch_prefix, min_gpus, gpu_count.
        """
        if not requires:
            return None

        arch_prefix = requires.get("arch_prefix")
        if arch_prefix and (not gfx_version or not gfx_version.startswith(arch_prefix)):
            return f"requires arch {arch_prefix}* (detected {gfx_version or 'unknown'})"

        min_gpus = requires.get("min_gpus")
        if min_gpus is not None and num_gpus < min_gpus:
            return f"requires {min_gpus}+ GPUs (have {num_gpus})"

        gpu_count = requires.get("gpu_count")
        if gpu_count is not None and num_gpus != gpu_count:
            return f"requires exactly {gpu_count} GPUs (have {num_gpus})"

        return None

    def run_tests(self) -> None:
        """Run RBT tests using the pre-built binary, save results to JSON."""
        log.info(f"Running {self.display_name}")

        # Get GPU info
        num_gpus, gfx_version = self.get_gpu_architecture()
        log.info(f"Detected {num_gpus} GPU(s), arch: {gfx_version}")

        # Locate pre-built RBT binary (built by TheRock build system)
        self.rbt_binary = self.rocm_path / "bin" / "rocm-bandwidth-test"
        if not self.rbt_binary.exists():
            raise TestExecutionError(
                f"RBT binary not found at {self.rbt_binary}\n"
                "Ensure TheRock was built with THEROCK_ENABLE_ROCM_BANDWIDTH_TEST=ON"
            )
        log.info(f"Using RBT binary: {self.rbt_binary}")

        # Setup environment - RBT needs ROCM_PATH to find plugins
        rbt_lib = self.rocm_path / "lib"
        env = self.get_rocm_env(
            additional_paths=[rbt_lib] if rbt_lib.exists() else None
        )
        env["ROCM_PATH"] = str(self.rocm_path)
        env["PATH"] = f"{self.rocm_path / 'bin'}:{env.get('PATH', '')}"

        # Get tests to run based on test type (from config)
        tests_to_run = self.tests_by_type.get(self.test_type) or self.tests_by_type.get(
            "full", []
        )
        log.info(f"Test type: {self.test_type.upper()}, Tests: {len(tests_to_run)}")

        # Run tests and collect results
        all_results = []

        for test_name in tests_to_run:
            # Get test definition from config
            test_def = self.test_definitions.get(test_name, {})
            cmd_str = test_def.get("cmd", "")
            cmd = [str(self.rbt_binary)] + shlex.split(cmd_str)
            requires = test_def.get("requires", {})
            extra_env = test_def.get("env")

            # Check requirements (all driven by config, no hardcoded arch names)
            skip_reason = self._get_skip_reason(requires, num_gpus, gfx_version)

            if skip_reason:
                log.info(f"SKIP: {test_name} - {skip_reason}")
                all_results.append(
                    {
                        "test_name": test_name,
                        "status": "SKIP",
                        "reason": skip_reason,
                    }
                )
                continue

            # Setup test environment
            test_env = env.copy()
            if extra_env:
                test_env.update(extra_env)

            # Run test and check exit code
            # Note: RBT doesn't support JSON/CSV output for test results.
            # For functional verification, exit code check is sufficient.
            return_code, output = self.execute_command(
                cmd, env=test_env, timeout=300, stream=True
            )
            passed = return_code == 0

            log.info(f"{'PASS' if passed else 'FAIL'}: {test_name}")

            all_results.append(
                {
                    "test_name": test_name,
                    "status": "PASS" if passed else "FAIL",
                    "reason": "" if passed else "Test execution failed",
                }
            )

        # Write results to JSON
        with open(self.results_json, "w") as f:
            json.dump(all_results, f, indent=2)

        log.info(f"Results saved to {self.results_json}")
        log.info(f"{self.display_name} execution complete")

    def parse_results(self) -> List[Dict[str, Any]]:
        """Parse RBT test results from JSON file.

        Returns:
            List of test result dictionaries
        """
        log.info(f"Parsing {self.display_name} Results")

        try:
            with open(self.results_json, "r") as f:
                json_results = json.load(f)
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

        test_results = []
        for result in json_results:
            test_results.append(
                self.create_test_result(
                    test_name=self.test_name,
                    subtest_name=result["test_name"],
                    status=result["status"],
                    suite="RBT",
                    reason=result.get("reason", ""),
                )
            )

        return test_results


if __name__ == "__main__":
    run_functional_main(RBTTest())
