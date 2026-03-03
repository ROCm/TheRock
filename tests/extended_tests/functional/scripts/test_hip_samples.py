"""
HIP Samples Functional Test

Tests HIP sample applications to ensure correct functionality across different
GPU architectures and sample types.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.logger import log
from utils.exceptions import TestExecutionError


class HipSamplesTest(FunctionalBase):
    """HIP Samples functional test."""

    def __init__(self):
        super().__init__(test_name="hip_samples", display_name="HIP Samples")

        self.results_json = self.script_dir / "hip_samples_results.json"

        # Load test configurations from JSON
        config = self.load_config("hip_samples.json")

        # Source directory for building from source
        self.rocm_systems_dir = self.therock_dir / "rocm-systems"
        self.hip_tests_samples_dir = (
            self.rocm_systems_dir / "projects" / "hip-tests" / "samples"
        )
        self.hip_tests_build_dir = (
            self.rocm_systems_dir / "projects" / "hip-tests" / "build"
        )

        # Store test execution results
        self.test_results = []

        # Load configuration from JSON
        self.skip_conditions = config.get("skip_conditions", {})
        self.skip_executables = config.get("skip_executables", [])
        self.special_compiler_cases = config.get("special_compiler_cases", {})

    def _build_sample(self, test_suite: str, testname: str) -> None:
        """Build a single HIP sample in its own build directory.

        Builds in {suite}/{testname}/build/ directory, matching the original pattern.
        """
        # Handle nested test cases (e.g., "15_static_library/device_functions")
        # testname may contain path separators
        source_dir = self.hip_tests_samples_dir / test_suite / testname
        build_dir = self.hip_tests_build_dir / test_suite / testname / "build"

        if not source_dir.exists():
            log.warning(f"Source directory not found: {source_dir}, skipping build")
            return

        # Check if CMakeLists.txt exists
        if not (source_dir / "CMakeLists.txt").exists():
            log.warning(f"CMakeLists.txt not found in {source_dir}, skipping build")
            return

        # Create build directory for this sample
        build_dir.mkdir(parents=True, exist_ok=True)

        # Get ROCm environment
        env = self.get_rocm_env()

        # Set ROCM_PATH and HIP_PATH environment variables for CMake HIP compiler detection
        # CMakeDetermineHIPCompiler.cmake checks these environment variables before CMakeLists.txt processes -DROCM_PATH
        rocm_path_str = str(self.rocm_path)
        env["ROCM_PATH"] = rocm_path_str
        env["HIP_PATH"] = rocm_path_str
        env["HIP_PLATFORM"] = "amd"
        # HIP_DEVICE_LIB_PATH helps CMake find ROCm device libraries
        env["HIP_DEVICE_LIB_PATH"] = str(
            self.rocm_path / "lib" / "llvm" / "amdgcn" / "bitcode"
        )
        # Ensure hipcc is in PATH for CMake HIP compiler detection
        rocm_bin = str(self.rocm_path / "bin")
        if "PATH" in env:
            env["PATH"] = f"{rocm_bin}:{env['PATH']}"
        else:
            env["PATH"] = rocm_bin

        # Build CMake command - default uses amdclang++, special cases override CXX/FC
        source_dir_str = str(source_dir)
        # Use self.rocm_path directly instead of hipconfig -l, which may return /opt/rocm
        amdclang_path = self.rocm_path / "lib" / "llvm" / "bin" / "amdclang++"

        # Set ROCM_PATH and CMAKE_PREFIX_PATH so CMake can find ROCm components
        # This is needed when /opt/rocm symlink is not present
        # Use -C to set initial cache values before project() call processes them
        rocm_path_str = str(self.rocm_path)
        cmake_cmd = [
            "cmake",
            f"-DCMAKE_CXX_COMPILER={amdclang_path}",
            f"-DROCM_PATH:STRING={rocm_path_str}",
            f"-DCMAKE_PREFIX_PATH:STRING={rocm_path_str};{rocm_path_str}/lib/llvm;{rocm_path_str}/hip",
            source_dir_str,
        ]

        # Handle special cases that need different compilers (from JSON config)
        if testname in self.special_compiler_cases:
            special_case = self.special_compiler_cases[testname]
            cxx_compiler = special_case.get("cxx_compiler", "amdclang++")
            fortran_compiler = special_case.get("fortran_compiler")

            if cxx_compiler == "clang++":
                cxx_path = self.rocm_path / "lib" / "llvm" / "bin" / "clang++"
            else:
                cxx_path = amdclang_path

            rocm_path_str = str(self.rocm_path)
            cmake_cmd = [
                "cmake",
                f'CXX="{cxx_path}"',
                f"-DROCM_PATH:STRING={rocm_path_str}",
                f"-DCMAKE_PREFIX_PATH:STRING={rocm_path_str};{rocm_path_str}/lib/llvm;{rocm_path_str}/hip",
                source_dir_str,
            ]

            if fortran_compiler:
                gfortran_result = subprocess.run(
                    ["which", fortran_compiler], capture_output=True, text=True
                )
                gfortran_path = (
                    gfortran_result.stdout.strip()
                    if gfortran_result.returncode == 0
                    else fortran_compiler
                )
                cmake_cmd.insert(2, f"FC={gfortran_path}")

        # Run CMake configure
        return_code = self.execute_command(cmake_cmd, cwd=build_dir, env=env)
        if return_code != 0:
            raise TestExecutionError(
                f"CMake configure failed for {test_suite}/{testname}"
            )

        # Build this sample
        make_cmd = ["make"]
        return_code = self.execute_command(make_cmd, cwd=build_dir, env=env)
        if return_code != 0:
            raise TestExecutionError(f"Build failed for {test_suite}/{testname}")

    def _initialize_build_environment(self) -> None:
        """Initialize build environment for HIP samples.

        Sets up the build directory structure and ensures submodules are initialized.
        Actual building happens per-sample in run_tests().
        """
        log.info("Initializing HIP samples build environment")

        # Initialize rocm-systems submodules if needed
        if not self.rocm_systems_dir.exists():
            raise TestExecutionError(
                f"rocm-systems directory not found at {self.rocm_systems_dir}\n"
                f"Ensure rocm-systems is present in TheRock directory"
            )

        # First, ensure rocm-systems submodule itself is initialized
        log.info("Checking rocm-systems submodule status")
        rocm_systems_submodule_cmd = [
            "git",
            "submodule",
            "update",
            "--init",
            "rocm-systems",
        ]
        return_code = self.execute_command(
            rocm_systems_submodule_cmd, cwd=self.therock_dir
        )
        if return_code != 0:
            log.warning(
                "Failed to initialize rocm-systems submodule, continuing anyway"
            )

        # Then, initialize only hip-tests submodule within rocm-systems
        # This is more efficient than updating all rocm-systems submodules
        hip_tests_path = "projects/hip-tests"
        log.info(f"Checking and initializing hip-tests submodule in rocm-systems")

        # Try to update only the hip-tests submodule path
        submodule_cmd = [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            hip_tests_path,
        ]
        return_code = self.execute_command(submodule_cmd, cwd=self.rocm_systems_dir)

        # If the specific path doesn't work (hip-tests might not be a submodule),
        # fall back to checking if it exists or needs initialization
        if return_code != 0:
            # Check if hip-tests exists as a directory (might be part of rocm-systems directly)
            hip_tests_dir = self.rocm_systems_dir / hip_tests_path
            if hip_tests_dir.exists():
                log.info("hip-tests directory exists (not a submodule)")
            else:
                # Try recursive update from rocm-systems root as fallback
                log.info(
                    "Attempting to initialize all rocm-systems submodules as fallback"
                )
                fallback_cmd = ["git", "submodule", "update", "--init", "--recursive"]
                return_code = self.execute_command(
                    fallback_cmd, cwd=self.rocm_systems_dir
                )
                if return_code != 0:
                    log.warning(
                        "Failed to initialize rocm-systems submodules, continuing anyway"
                    )

        if not self.hip_tests_samples_dir.exists():
            raise TestExecutionError(
                f"HIP samples source not found at {self.hip_tests_samples_dir}\n"
                f"Ensure rocm-systems repository is cloned correctly"
            )

        # Clean build directory if it exists
        if self.hip_tests_build_dir.exists():
            log.info(f"Removing existing build directory: {self.hip_tests_build_dir}")
            shutil.rmtree(self.hip_tests_build_dir)

        # Create build directory
        self.hip_tests_build_dir.mkdir(parents=True, exist_ok=True)

        log.info("Build environment initialized")

    def _discover_test_structure(self) -> Dict[str, List[str]]:
        """Discover test structure from samples directory.

        Handles nested test cases (e.g., 15_static_library/device_functions).
        Only includes directories that have CMakeLists.txt or contain subdirectories with CMakeLists.txt.

        Returns:
            Dictionary mapping test suite names to lists of test case names (may include paths like "15_static_library/device_functions")
        """
        test_structure = {}

        if not self.hip_tests_samples_dir.exists():
            return test_structure

        def discover_test_cases(suite_dir: Path, base_path: Path = None) -> List[str]:
            """Recursively discover test cases in a suite directory."""
            test_cases = []
            if base_path is None:
                base_path = suite_dir

            for test_dir in sorted(suite_dir.iterdir()):
                if not test_dir.is_dir():
                    continue

                # Skip build directories
                if test_dir.name == "build":
                    continue

                # Check if this directory has CMakeLists.txt (it's a test case)
                if (test_dir / "CMakeLists.txt").exists():
                    # Relative path from suite directory
                    rel_path = test_dir.relative_to(base_path)
                    test_cases.append(str(rel_path))
                else:
                    # Check if subdirectories have CMakeLists.txt (nested test cases)
                    nested_cases = discover_test_cases(test_dir, base_path)
                    if nested_cases:
                        test_cases.extend(nested_cases)
                    # If no nested cases found, but directory exists, might be a parent dir
                    # Skip it if it doesn't have CMakeLists.txt

            return test_cases

        # Discover test suites (directories in samples/)
        for suite_dir in sorted(self.hip_tests_samples_dir.iterdir()):
            if not suite_dir.is_dir():
                continue

            suite_name = suite_dir.name
            test_cases = discover_test_cases(suite_dir)

            if test_cases:
                test_structure[suite_name] = test_cases

        return test_structure

    def _is_cmake_internal_executable(self, exec_path: Path) -> bool:
        """Check if an executable is a CMake internal file that should be skipped.

        CMake creates internal test executables during configuration that should not be run.
        """
        exec_name = exec_path.name
        exec_path_str = str(exec_path)

        # Skip CMake internal executables
        cmake_patterns = [
            "CMakeDetermineCompiler",
            "CMakeTest",
            "cmake_",
            "CMakeCCompiler",
            "CMakeCXXCompiler",
        ]

        # Check if path contains CMakeFiles directory
        if "CMakeFiles" in exec_path_str:
            return True

        # Check if name matches CMake patterns
        for pattern in cmake_patterns:
            if pattern in exec_name:
                return True

        return False

    def _should_skip_executable(self, exec_path: Path) -> bool:
        """Check if an executable should be skipped (not run).

        Some executables require command-line arguments or are not meant to be run directly.
        """
        exec_name = exec_path.name

        # Skip object files (.o files) - these are not executables
        if exec_name.endswith(".o"):
            return True

        # Skip executables from JSON config
        if exec_name in self.skip_executables:
            return True

        return False

    def _find_executables_in_build(self, test_suite: str, testname: str) -> List[Path]:
        """Find all executables for a test case in the build directory.

        Since each sample is built in its own directory, we search in:
        - build_dir/{suite}/{testname}/build/ (where the sample is built)
        - Install location (CMAKE_INSTALL_PREFIX/bin)

        Filters out CMake internal executables.

        Returns:
            List of executable paths found (excluding CMake internals)
        """
        executables = []
        found_paths = set()  # Avoid duplicates

        # Primary location: individual build directory for this sample
        # Build happens in {suite}/{testname}/build/ directory
        sample_build_dir = self.hip_tests_build_dir / test_suite / testname / "build"

        # Search locations (in order of preference)
        search_paths = [
            # Individual sample build directory (where we build each sample)
            sample_build_dir,
            # Also check parent directory in case executables are there
            self.hip_tests_build_dir / test_suite / testname,
            # Install location (CMAKE_INSTALL_PREFIX/bin)
            Path(self.rocm_path) / "bin",
        ]

        # Search in each location
        for search_path in search_paths:
            if not search_path.exists():
                continue

            # For build directory, first check top-level (where actual executables are)
            # Then do recursive search but skip CMakeFiles directories
            try:
                if search_path == sample_build_dir:
                    # First, check top-level files (most common case)
                    for item in search_path.iterdir():
                        if not item.is_file():
                            continue

                        # Skip CMake internal executables
                        if self._is_cmake_internal_executable(item):
                            continue

                        # Skip executables that shouldn't be run
                        if self._should_skip_executable(item):
                            continue

                        # Check if executable
                        if os.access(item, os.X_OK):
                            if item not in found_paths:
                                executables.append(item)
                                found_paths.add(item)

                    # Then do recursive search, but skip CMakeFiles directories entirely
                    for item in search_path.rglob("*"):
                        # Skip CMakeFiles directories entirely
                        if "CMakeFiles" in str(item):
                            continue

                        if not item.is_file():
                            continue

                        # Skip CMake internal executables (extra safety check)
                        if self._is_cmake_internal_executable(item):
                            continue

                        # Skip executables that shouldn't be run
                        if self._should_skip_executable(item):
                            continue

                        # Check if executable
                        if os.access(item, os.X_OK):
                            if item not in found_paths:
                                executables.append(item)
                                found_paths.add(item)
                else:
                    # For other locations (install, etc.), do recursive search with filtering
                    for item in search_path.rglob("*"):
                        # Skip CMakeFiles directories entirely
                        if "CMakeFiles" in str(item):
                            continue

                        if not item.is_file():
                            continue

                        # Skip CMake internal executables
                        if self._is_cmake_internal_executable(item):
                            continue

                        # Skip executables that shouldn't be run
                        if self._should_skip_executable(item):
                            continue

                        # Check if executable
                        if os.access(item, os.X_OK):
                            # For install location, match by name pattern
                            if (
                                testname.lower() in item.name.lower()
                                or item.name.lower() in testname.lower()
                            ):
                                if item not in found_paths:
                                    executables.append(item)
                                    found_paths.add(item)
            except (PermissionError, OSError) as e:
                continue

        return executables

    def _run_sample(self, test_suite: str, testname: str) -> Dict[str, Any]:
        """Run a single HIP sample and return result.

        Executables are already built from the top-level build, so we find and run them.
        """
        log.info(f"Running {test_suite}/{testname}")

        # Find all executables for this test case
        executables = self._find_executables_in_build(test_suite, testname)

        if not executables:
            log.warning(f"No executables found for {test_suite}/{testname}")
            result = {
                "test_suite": test_suite,
                "test_case": testname,
                "command": "",
                "return_code": -1,
                "status": "FAIL",
                "error": f"No executables found in build directory",
            }
            log.info(f"Completed {test_suite}/{testname} - Status: FAIL")
            return result

        env = self.get_rocm_env()
        error_message = None
        exec_names = []
        overall_return_code = 0  # Track overall status across all executables
        captured_output = []  # Capture output for validation

        # Run each executable found
        for exec_path in executables:
            exec_names.append(exec_path.name)
            cmd = [str(exec_path)]
            return_code, output = self._execute_command_with_output(
                cmd, cwd=exec_path.parent, env=env
            )
            captured_output.append(output)

            # If any executable fails, mark overall as failed
            if return_code != 0:
                overall_return_code = return_code
                error_message = f"Execution failed with return code {return_code}"
                break

        # Parse output to determine status based on return code and output
        status = self._parse_sample_output(
            test_suite, testname, overall_return_code, "\n".join(captured_output)
        )

        result = {
            "test_suite": test_suite,
            "test_case": testname,
            "command": " ".join(exec_names),
            "return_code": overall_return_code,
            "status": status,
        }
        if error_message:
            result["error"] = error_message

        # Log test completion with status
        log.info(f"Completed {test_suite}/{testname} - Status: {status}")

        return result

    def _execute_command_with_output(
        self, cmd: List[str], cwd: Path = None, env: Dict[str, str] = None
    ) -> Tuple[int, str]:
        """Execute a command and capture output for validation.

        Returns:
            Tuple of (return_code, output_text)
        """
        work_dir = cwd or self.therock_dir
        log.info(f"++ Exec [{work_dir}]$ {shlex.join(cmd)}")

        # Merge custom env with current environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
        )

        output_lines = []
        for line in process.stdout:
            line_text = line.strip()
            log.info(line_text)
            output_lines.append(line_text)

        process.wait()
        return process.returncode, "\n".join(output_lines)

    def _parse_sample_output(
        self, test_suite: str, testname: str, return_code: int, output: str = ""
    ) -> str:
        """Parse sample output to determine test status.

        Args:
            test_suite: Test suite name
            testname: Test case name
            return_code: Process return code
            output: Captured stdout/stderr output (unused, kept for potential future use)

        Returns:
            Status string: "PASS", "FAIL", "ERROR", or "SKIP"
        """
        # Exit code is the primary and reliable indicator of test status
        if return_code != 0:
            return "FAIL"

        return "PASS"

    def _should_skip_test(self, test_suite: str, testname: str) -> bool:
        """Check if a test should be skipped based on JSON configuration."""
        if testname not in self.skip_conditions:
            return False

        skip_config = self.skip_conditions[testname]

        # Check GPU architecture requirement
        if "requires_gpu" in skip_config:
            gfx_id = self.get_gpu_architecture()
            required_gpu = skip_config["requires_gpu"]
            if gfx_id != required_gpu:
                log.info(
                    f"Skipping {testname} - requires {required_gpu}, current GPU: {gfx_id}"
                )
                return True

        # Check GPU count requirement
        if "requires_gpu_count" in skip_config:
            gpu_count = self.get_gpu_count()
            required_count = skip_config["requires_gpu_count"]
            if gpu_count < required_count:
                log.info(
                    f"Skipping {testname} - requires {required_count} GPUs, current GPU count: {gpu_count}"
                )
                return True

        # Check tool requirement
        if "requires_tool" in skip_config:
            tool_name = skip_config["requires_tool"]
            tool_paths = skip_config.get("tool_paths", [])

            # Check in ROCm path first
            rocm_tool_path = self.rocm_path / "llvm" / "bin" / tool_name
            if rocm_tool_path.exists():
                return False

            # Check configured paths
            for tool_path_str in tool_paths:
                tool_path = Path(tool_path_str)
                if tool_path.exists():
                    return False

            log.info(
                f"Skipping {testname} - {tool_name} not found (requires {skip_config.get('description', 'tool')})"
            )
            return True

        return False

    def run_tests(self) -> None:
        """Run HIP samples tests and save results to JSON.

        Uses interleaved approach: build one sample, test it, then move to next.
        This provides faster feedback and better error handling.
        """
        log.info(f"Running {self.display_name} Tests")

        # Initialize build environment
        self._initialize_build_environment()

        # Verify samples source exists
        if not self.hip_tests_samples_dir.exists():
            raise TestExecutionError(
                f"HIP samples source not found at {self.hip_tests_samples_dir}\n"
                f"Ensure rocm-systems repository is set up correctly"
            )

        # Detect GPU architecture
        gfx_id = self.get_gpu_architecture()
        log.info(f"Detected GPU architecture: {gfx_id}")

        # Discover test structure from samples directory
        test_structure = self._discover_test_structure()

        if not test_structure:
            raise TestExecutionError(
                f"No test suites found in {self.hip_tests_samples_dir}\n"
                f"Ensure samples directory structure is correct"
            )

        log.info(f"Discovered {len(test_structure)} test suites")

        # Build and test each sample together (interleaved approach)
        for test_suite, test_cases in test_structure.items():
            log.info(
                f"Processing test suite: {test_suite} ({len(test_cases)} test cases)"
            )

            for testname in test_cases:
                # Check if should skip
                if self._should_skip_test(test_suite, testname):
                    self.test_results.append(
                        {
                            "test_suite": test_suite,
                            "test_case": testname,
                            "command": "",
                            "return_code": 0,
                            "status": "SKIP",
                        }
                    )
                    continue

                # Build and test this sample
                try:
                    # Build the sample
                    log.info(f"Building {test_suite}/{testname}")
                    self._build_sample(test_suite, testname)

                    # Immediately test the sample we just built
                    result = self._run_sample(test_suite, testname)
                    self.test_results.append(result)

                except TestExecutionError as e:
                    # Build or execution failed
                    log.error(f"Failed to build/test {test_suite}/{testname}: {e}")
                    error_result = {
                        "test_suite": test_suite,
                        "test_case": testname,
                        "command": "",
                        "return_code": -1,
                        "status": "ERROR",
                        "error": str(e),
                    }
                    self.test_results.append(error_result)
                except Exception as e:
                    # Unexpected error
                    log.error(f"Unexpected error for {test_suite}/{testname}: {e}")
                    error_result = {
                        "test_suite": test_suite,
                        "test_case": testname,
                        "command": "",
                        "return_code": -1,
                        "status": "ERROR",
                        "error": str(e),
                    }
                    self.test_results.append(error_result)

        # Write results to JSON
        with open(self.results_json, "w") as f:
            json.dump(self.test_results, f, indent=2)

        log.info(f"{self.display_name} results saved to {self.results_json}")
        log.info(f"{self.display_name} test execution complete")

    def parse_results(self) -> List[Dict[str, Any]]:
        """Parse test results from JSON file.

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
            status = result.get("status", "ERROR")

            # Ensure status is uppercase and valid
            if status.upper() not in ["PASS", "FAIL", "ERROR", "SKIP"]:
                log.warning(
                    f"Invalid status '{status}' for {result.get('test_case')}, defaulting to ERROR"
                )
                status = "ERROR"
            else:
                status = status.upper()

            test_results.append(
                self.create_test_result(
                    test_name=self.test_name,
                    subtest_name=result["test_case"],
                    status=status,
                    suite=result["test_suite"],
                    command=result.get("command", ""),
                )
            )

        log.info(f"Parsed {len(test_results)} test results")
        return test_results


if __name__ == "__main__":
    run_functional_main(HipSamplesTest())
