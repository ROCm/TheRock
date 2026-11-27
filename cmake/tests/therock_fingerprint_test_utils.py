# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared setup for the cmake tests"""

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
ProjectFingerprints = dict[str, str]


def write(path: Path, contents: str) -> None:
    """Write an indented fixture, creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def cmake_project_setup(project_name: str) -> str:
    """Initialize the TheRock modules needed by a fixture's subprojects."""
    return f"""
        cmake_minimum_required(VERSION 3.25)
        project({project_name} NONE)
        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")
        include(therock_globals)
        include(therock_sanitizers)
        include(therock_flag_utils)
        include(therock_default_targets)
        include(therock_subproject)
    """


class FingerprintTestCase(unittest.TestCase):
    """Provide tool checks, a temporary directory, and isolated command execution."""

    required_tools: tuple[str, ...] = ()
    command_timeout_seconds = 180

    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name in cls.required_tools if not shutil.which(name)]
        if missing:
            raise RuntimeError(
                f"Required tools unavailable: {', '.join(missing)}; tests were not run"
            )

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="therock-fingerprint-")
        self.addCleanup(temporary.cleanup)
        # Resolve Windows 8.3 temporary paths before passing them to CMake.
        self.root = Path(temporary.name).resolve()
        self.env = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("GIT_")
        }
        self.env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_AUTHOR_NAME="Fingerprint test",
            GIT_AUTHOR_EMAIL="test@example.invalid",
            GIT_COMMITTER_NAME="Fingerprint test",
            GIT_COMMITTER_EMAIL="test@example.invalid",
        )

    def run_command(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        """Run a fixture command and include its output in any failure."""
        result = subprocess.run(
            [str(arg) for arg in args],
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.command_timeout_seconds,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Command failed: {args}\n{result.stdout}\n{result.stderr}",
        )
        return result
