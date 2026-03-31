# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared base class for benchmark and functional tests.

Both BenchmarkBase and FunctionalBase inherit from ExtendedTestBase.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Union

# Add build_tools/github_actions to path for github_actions_api
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "build_tools" / "github_actions")
)

from .logger import log
from .exceptions import TestExecutionError
from .extended_test_client import ExtendedTestClient

from github_actions_api import (
    gha_append_step_summary,
)


class ExtendedTestBase:
    """Base class providing shared infrastructure for extended tests."""

    def __init__(self, test_name: str, display_name: str = None):
        """Initialize common extended test infrastructure.

        Args:
            test_name: Internal test name (e.g., 'rocfft', 'miopen_driver_conv')
            display_name: Display name for reports, defaults to test_name
        """
        self.test_name = test_name
        self.display_name = display_name or test_name

        # Environment variables
        self.therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
        self.artifact_run_id = os.getenv("ARTIFACT_RUN_ID")
        self.amdgpu_families = os.getenv("AMDGPU_FAMILIES")
        self.therock_dir = Path(__file__).resolve().parents[3]
        self.rocm_path = (
            Path(self.therock_bin_dir).resolve().parent
            if self.therock_bin_dir
            else None
        )

        # Initialize test client with auto-detection
        self.client = ExtendedTestClient(auto_detect=True)
        self.client.print_system_summary()

    def load_config(self, config_filename: str) -> Dict[str, Any]:
        """Load test configuration from JSON file in configs/ directory.

        Expects self.script_dir to be set by the child class.
        Config file is resolved as: <script_dir>/../configs/<config_filename>
        """
        config_file = self.script_dir.parent / "configs" / config_filename

        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise TestExecutionError(
                f"Configuration file not found: {config_file}\n"
                f"Ensure {config_filename} exists in configs/ directory"
            )
        except json.JSONDecodeError as e:
            raise TestExecutionError(
                f"Invalid JSON in configuration file: {e}\n"
                f"Check JSON syntax in {config_filename}"
            )

    def execute_command(
        self,
        cmd: List[str],
        log_file_handle: Optional[IO] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        capture_output: bool = False,
    ) -> Union[int, subprocess.CompletedProcess[str]]:
        """Execute a command, stream output to log, and optionally write to a file.

        Args:
            cmd: Command list to execute.
            log_file_handle: Optional file handle to write output to (used by benchmarks).
            env: Optional environment variables to merge with current environment.
            cwd: Working directory for the command. Defaults to self.therock_dir.
            timeout: Optional timeout in seconds. Returns code 124 when exceeded.
            capture_output: If True, return subprocess.CompletedProcess with captured stdout/stderr.

        Returns:
            Exit code from the command when capture_output is False.
            subprocess.CompletedProcess when capture_output is True.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be > 0 when provided, got: {timeout}")

        timeout_exit_code = 124
        run_cwd = cwd or str(self.therock_dir)
        log.info(f"++ Exec [{run_cwd}]$ {shlex.join(cmd)}")
        if log_file_handle:
            log_file_handle.write(f"{shlex.join(cmd)}\n")

        # Merge custom env with current environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=run_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_output else subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        stdout_data = ""
        stderr_data = "" if capture_output else ""
        timed_out = False

        try:
            stdout_data, stderr_data = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            log.error(f"Command timed out after {timeout} seconds: {shlex.join(cmd)}")
            process.kill()
            cleanup_stdout = ""
            cleanup_stderr = ""
            try:
                cleanup_stdout, cleanup_stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                # Return partial output if pipes still have unread data.
                cleanup_stdout = ""
                cleanup_stderr = ""

            stdout_data = f"{exc.stdout or ''}{cleanup_stdout or ''}"
            if capture_output:
                stderr_data = f"{exc.stderr or ''}{cleanup_stderr or ''}"

        output_for_log = stdout_data
        if capture_output and stderr_data:
            output_for_log = f"{output_for_log}{stderr_data}"

        for line in output_for_log.splitlines(True):
            log.info(line.rstrip())
            if log_file_handle:
                log_file_handle.write(line)

        rc = timeout_exit_code if timed_out else process.returncode
        if capture_output:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=rc,
                stdout=stdout_data,
                stderr=stderr_data,
            )

        return rc

    def create_test_result(
        self,
        test_name: str,
        subtest_name: str,
        status: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a standardized test result dictionary.

        Builds the base result structure used by extended tests.

        Args:
            test_name: Test name (e.g., 'rocfft')
            subtest_name: Specific test/suite identifier
            status: 'PASS', 'FAIL', 'ERROR', or 'SKIP'
            **kwargs: Additional fields (e.g., score/unit/flag/batch_size/ngpu for benchmarks,
                suite/command for functional tests).
        """
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        )

        # Build test_config with base fields + all kwargs
        test_config = {
            "test_name": test_name,
            "sub_test_name": subtest_name,
            "python_version": python_version,
            "environment_dependencies": [],
            **kwargs,
        }

        return {
            "test_name": test_name,
            "subtest": subtest_name,
            "status": status,
            "test_config": test_config,
            **kwargs,
        }

    def calculate_statistics(
        self,
        test_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate pass/fail statistics from test results and return a dictionary with the counts.

        Args:
            test_results: List of test result dicts with 'status' key.
        """
        counts = {
            "passed": sum(1 for r in test_results if r.get("status") == "PASS"),
            "failed": sum(1 for r in test_results if r.get("status") == "FAIL"),
            "error": sum(1 for r in test_results if r.get("status") == "ERROR"),
            "skipped": sum(1 for r in test_results if r.get("status") == "SKIP"),
        }

        counts["total"] = len(test_results)
        counts["overall_status"] = (
            "PASS" if (counts["failed"] + counts["error"]) == 0 else "FAIL"
        )

        return counts

    def upload_results(
        self,
        test_results: List[Dict[str, Any]],
        stats: Dict[str, Any],
        test_type: str,
        output_dir: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upload test results to API and save locally.

        Args:
            test_results: List of test result dictionaries.
            stats: Statistics from calculate_statistics().
            test_type: 'benchmark' or 'functional'.
            output_dir: Directory to save local results.
            extra_metadata: Optional additional metadata for upload.
        """
        metadata = {
            "artifact_run_id": self.artifact_run_id,
            "amdgpu_families": self.amdgpu_families,
            "test_name": self.test_name,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        log.info(f"Uploading {test_type.title()} Test Results to API")
        success = self.client.upload_results(
            test_name=f"{self.test_name}_{test_type}",
            test_results=test_results,
            test_status=stats["overall_status"],
            test_metadata=metadata,
            save_local=True,
            output_dir=output_dir,
        )

        if success:
            log.info("Results uploaded successfully")
        else:
            log.error(
                f"API upload failed for '{self.test_name}' {test_type} results. "
                f"Results were saved locally to '{output_dir}'. "
            )

        return success

    def get_rocm_env(self, additional_paths: List[Path] = None) -> Dict[str, str]:
        """Get environment with LD_LIBRARY_PATH set for ROCm libraries.

        Args:
            additional_paths: Additional library paths to include

        Returns:
            Environment dictionary with LD_LIBRARY_PATH configured
        """
        env = os.environ.copy()
        rocm_lib = self.rocm_path / "lib"

        # Build list of library paths
        lib_paths = [str(rocm_lib)]
        if additional_paths:
            lib_paths.extend(str(p) for p in additional_paths)

        # Append existing LD_LIBRARY_PATH if present
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            lib_paths.append(existing)

        env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
        return env
