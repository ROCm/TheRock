#!/usr/bin/env python3
"""
ROCm Bandwidth Test (RBT) Test.

Clones, builds and executes ROCm Bandwidth Test to validate GPU memory bandwidth
and peer-to-peer communication functionality.
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

        # RBT directory paths
        self.rbt_dir = self.rocm_path / "bin" / "rbt"
        self.rbt_build_dir = self.rbt_dir / "build"
        self.rbt_install_dir = self.rbt_build_dir / "install"

        # Load test configuration from JSON
        config = self.load_config("rbt.json")
        self.git_url = config["git_url"]
        self.git_branch = config["git_branch"]

        # Test configuration (env var overrides config)
        self.test_type = os.getenv("TEST_TYPE", config.get("test_type", "full"))
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"

        # Test definitions from config
        self.tests_by_type = config.get("tests", {})
        self.test_definitions = config.get("test_definitions", {})

    def _build_rbt(self) -> None:
        """Build ROCm Bandwidth Test."""
        log.info("Building ROCm Bandwidth Test")

        # Setup environment for build
        env = self.get_rocm_env()
        env["HIP_PLATFORM"] = "amd"
        env["ROCM_PATH"] = str(self.rocm_path)
        env["HIP_PATH"] = str(self.rocm_path)
        env["HIP_CLANG_PATH"] = str(self.rocm_path / "lib" / "llvm" / "bin")

        # Device library path for TheRock layout
        device_lib_path = self.rocm_path / "lib" / "llvm" / "amdgcn" / "bitcode"
        if device_lib_path.exists():
            env["HIP_DEVICE_LIB_PATH"] = str(device_lib_path)
        else:
            std_device_lib = self.rocm_path / "amdgcn" / "bitcode"
            if std_device_lib.exists():
                env["HIP_DEVICE_LIB_PATH"] = str(std_device_lib)

        # Add ROCm bin to PATH
        rocm_bin = str(self.rocm_path / "bin")
        llvm_bin = str(self.rocm_path / "lib" / "llvm" / "bin")
        env["PATH"] = f"{rocm_bin}:{llvm_bin}:{env.get('PATH', '')}"

        # Create build directory
        self.rbt_build_dir.mkdir(parents=True, exist_ok=True)

        # RPATH settings
        lib_rpath = "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../lib/llvm/lib"

        # CMake configure
        cmake_args = [
            "cmake",
            f"-DCMAKE_BUILD_TYPE=Release",
            "-DAMD_APP_STANDALONE_BUILD_PACKAGE=OFF",
            "-DAMD_APP_ROCM_BUILD_PACKAGE=ON",
            f"-DCMAKE_PREFIX_PATH={self.rocm_path}",
            f"-DROCM_PATH={self.rocm_path}",
            f"-DCMAKE_INSTALL_PREFIX={self.rbt_install_dir}",
            f'-DCMAKE_INSTALL_RPATH={lib_rpath}',
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_SKIP_BUILD_RPATH=OFF",
            "-DAMD_APP_BUILD_TESTS=OFF",
            "..",
        ]

        build_steps = [
            {
                "name": "cmake configure",
                "cmd": cmake_args,
                "cwd": self.rbt_build_dir,
            },
            {
                "name": "cmake build",
                "cmd": ["cmake", "--build", ".", f"-j{os.cpu_count()}"],
                "cwd": self.rbt_build_dir,
            },
            {
                "name": "cmake install",
                "cmd": ["cmake", "--install", "."],
                "cwd": self.rbt_build_dir,
            },
        ]

        with open(self.log_file, "a") as f:
            for step in build_steps:
                log.info(f"Running build step: {step['name']}")
                f.write(f"\n=== {step['name']} ===\n")

                return_code, _ = self.execute_command(
                    step["cmd"], cwd=step["cwd"], env=env, log_file_handle=f
                )

                if return_code != 0:
                    raise TestExecutionError(
                        f"RBT build failed at step '{step['name']}'\n"
                        f"Check {self.log_file} for details"
                    )

        log.info("RBT build completed")

    def run_tests(self) -> None:
        """Clone, build, and run RBT tests, save results to JSON."""
        log.info(f"Running {self.display_name}")

        # Check if RBT binary already exists in ROCm bin
        system_rbt = self.rocm_path / "bin" / "rocm-bandwidth-test"
        if system_rbt.exists():
            log.info(f"RBT binary found at {system_rbt}, skipping build")
            self.rbt_binary = system_rbt
            bin_path = self.rocm_path / "bin"
        else:
            log.info("RBT binary not found in ROCm, building from source")
            # Clone RBT repository and update submodules
            self.clone_repository(
                git_url=self.git_url,
                branch=self.git_branch,
                target_dir=self.rbt_dir,
                update_submodules=True,
            )

            # Build RBT
            self._build_rbt()

            # Set binary path
            self.rbt_binary = self.rbt_install_dir / "bin" / "rocm-bandwidth-test"
            bin_path = self.rbt_install_dir / "bin"

            if not self.rbt_binary.exists():
                raise TestExecutionError(
                    f"RBT binary not found at {self.rbt_binary}\n"
                    f"Build may have failed, check {self.log_file}"
                )

        # Setup environment with ROCm libraries
        env = self.get_rocm_env()
        env["PATH"] = f"{bin_path}:{env.get('PATH', '')}"

        # Get GPU info
        num_gpus, gfx_version = self.get_gpu_architecture()
        has_gfx94x = gfx_version.startswith("gfx94") if gfx_version else False
        log.info(f"Detected {num_gpus} GPU(s), arch: {gfx_version}")

        # Get tests to run based on test type (from config)
        rbt = str(self.rbt_binary)
        tests_to_run = self.tests_by_type.get(self.test_type) or self.tests_by_type.get("full", [])
        log.info(f"Test type: {self.test_type.upper()}, Tests: {len(tests_to_run)}")

        # Run tests and collect results
        all_results = []

        for test_name in tests_to_run:
            # Get test definition from config
            test_def = self.test_definitions.get(test_name, {})
            cmd_str = test_def.get("cmd", "")
            cmd = [rbt] + shlex.split(cmd_str)
            requirement = test_def.get("requires")
            extra_env = test_def.get("env")

            # Check requirements
            skip_reason = None
            if requirement == "multi_gpu" and num_gpus < 2:
                skip_reason = f"requires 2+ GPUs (have {num_gpus})"
            elif requirement == "gfx94x_8gpu":
                if not has_gfx94x:
                    skip_reason = "requires MI300 GPU"
                elif num_gpus != 8:
                    skip_reason = f"requires exactly 8 GPUs (have {num_gpus})"

            if skip_reason:
                log.info(f"SKIP: {test_name} - {skip_reason}")
                all_results.append({
                    "test_name": test_name,
                    "status": "SKIP",
                    "reason": skip_reason,
                })
                continue

            # Setup test environment
            test_env = env.copy()
            if extra_env:
                test_env.update(extra_env)

            # Run test and check exit code
            # Note: RBT doesn't support JSON/CSV output for test results.
            # For functional verification, exit code check is sufficient.
            return_code, output = self.execute_command(
                cmd, env=test_env, timeout=300, stream=False
            )
            passed = return_code == 0

            log.info(f"{'PASS' if passed else 'FAIL'}: {test_name}")
            if self.verbose and output:
                log.info(f"Output:\n{output[:1000]}")

            all_results.append({
                "test_name": test_name,
                "status": "PASS" if passed else "FAIL",
                "reason": "" if passed else "Test execution failed",
            })

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
