# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Base class for component test runners.

This module provides a base class that encapsulates common patterns for running
tests in TheRock components. Each component can subclass this and override
specific methods to customize behavior.

Common patterns abstracted:
- Environment variable handling (THEROCK_BIN_DIR, TEST_TYPE, SHARD_INDEX, etc.)
- GTest sharding setup
- CTest and GTest command construction
- Test filtering (quick vs full)
- Logging and command execution
"""

import logging
import os
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class BaseTestRunner(ABC):
    """
    Abstract base class for component test runners.

    Subclasses must implement:
    - build_command(): Construct the test command to execute

    Subclasses may override:
    - setup_environment(): Customize environment variables
    - get_test_directory(): Override the default test directory
    - validate_environment(): Add custom environment validation
    - get_quick_test_filters(): Define quick test patterns
    """

    def __init__(self, component_name: str):
        """
        Initialize the test runner.

        Args:
            component_name: Name of the component being tested (e.g., "rocrand", "rocsolver")
        """
        self.component_name = component_name

        # Standard environment variables
        self.therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
        self.script_dir = Path(__file__).resolve().parent
        self.therock_dir = self.script_dir.parent.parent.parent

        # Test configuration
        self.test_type = os.getenv("TEST_TYPE", "full")
        self.amdgpu_families = os.getenv("AMDGPU_FAMILIES")
        self.runner_os = os.getenv("RUNNER_OS", "").lower()

        # Sharding configuration
        self.shard_index = int(os.getenv("SHARD_INDEX", "1"))
        self.total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

        # Output directories
        self.output_artifacts_dir = os.getenv("OUTPUT_ARTIFACTS_DIR")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Environment variables for subprocess
        self.environ_vars = os.environ.copy()

    def is_quick_test(self) -> bool:
        """Check if running in quick test mode."""
        return self.test_type == "quick"

    def setup_gtest_sharding(self) -> None:
        """
        Configure GTest sharding environment variables.

        GTest uses 0-indexed sharding, but GitHub Actions uses 1-indexed arrays.
        This method converts and sets GTEST_SHARD_INDEX and GTEST_TOTAL_SHARDS.
        """
        self.environ_vars["GTEST_SHARD_INDEX"] = str(self.shard_index - 1)
        self.environ_vars["GTEST_TOTAL_SHARDS"] = str(self.total_shards)

    def setup_rocm_path(self) -> None:
        """
        Set ROCM_PATH environment variable for tests that require it.

        Some runtime kernel compilations rely on ROCM_PATH being set.
        This sets it to the parent directory of THEROCK_BIN_DIR.
        """
        if self.therock_bin_dir:
            rocm_path = Path(self.therock_bin_dir).resolve().parent
            self.environ_vars["ROCM_PATH"] = str(rocm_path)

    def setup_environment(self) -> None:
        """
        Setup environment variables for the test run.

        Default implementation sets up GTest sharding and ROCM_PATH.
        Subclasses can override to add component-specific environment setup.
        """
        self.setup_gtest_sharding()
        self.setup_rocm_path()

    def get_test_directory(self) -> str:
        """
        Get the test directory path.

        Default implementation returns THEROCK_BIN_DIR/component_name.
        Subclasses can override for custom test directory layouts.

        Returns:
            Path to the test directory
        """
        return f"{self.therock_bin_dir}/{self.component_name}"

    def validate_environment(self) -> None:
        """
        Validate required environment variables are set.

        Default implementation checks THEROCK_BIN_DIR.
        Subclasses can override to add additional validation.

        Raises:
            SystemExit: If required environment variables are missing
        """
        if not self.therock_bin_dir:
            self.logger.error("THEROCK_BIN_DIR environment variable is not set")
            sys.exit(1)

    def get_quick_test_filters(self) -> Optional[List[str]]:
        """
        Get quick test filter patterns.

        Subclasses should override this to provide component-specific quick test patterns.

        Returns:
            List of filter patterns for quick tests, or None if not applicable
        """
        return None

    @abstractmethod
    def build_command(self) -> List[str]:
        """
        Build the test command to execute.

        This method must be implemented by subclasses to construct the
        appropriate test command (ctest, gtest binary, etc.) with filters
        and options specific to the component.

        Returns:
            List of command arguments suitable for subprocess.run()
        """
        pass

    def run(self) -> int:
        """
        Execute the test runner.

        This is the main entry point that:
        1. Validates the environment
        2. Sets up environment variables
        3. Builds the test command
        4. Executes the command
        5. Returns the exit code

        Returns:
            Exit code from the test command (0 for success)
        """
        self.validate_environment()
        self.setup_environment()

        cmd = self.build_command()

        self.logger.info(f"++ Exec [{self.therock_dir}]$ {shlex.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.therock_dir,
            env=self.environ_vars,
            check=False
        )

        return result.returncode
