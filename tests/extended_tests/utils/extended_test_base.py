"""
Shared base class for benchmark and functional tests.

Both BenchmarkBase and FunctionalBase inherit from ExtendedTestBase.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

# Add build_tools/github_actions to path for github_actions_utils
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "build_tools" / "github_actions")
)

from .logger import log
from .exceptions import TestExecutionError
from .extended_test_client import ExtendedTestClient

from github_actions_utils import (
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

        # Initialize test client with auto-detection
        self.client = ExtendedTestClient(auto_detect=True)
        self.client.print_system_summary()

    def execute_command(
        self,
        cmd: List[str],
        log_file_handle: Optional[IO] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> int:
        """Execute a command, stream output to log, and optionally write to a file.

        Args:
            cmd: Command list to execute.
            log_file_handle: Optional file handle to write output to (used by benchmarks).
            env: Optional environment variables to merge with current environment.
            cwd: Working directory for the command. Defaults to self.therock_dir.

        Returns:
            Exit code from the command.
        """
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
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        for line in process.stdout:
            log.info(line.strip())
            if log_file_handle:
                log_file_handle.write(f"{line}")

        process.wait()
        return process.returncode

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
