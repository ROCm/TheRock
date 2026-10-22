#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``downstream_cmake_smoke.py``.

No ROCm install and no real ``cmake`` binary are required. Configure is
replaced with ``run_configure``, and ``shutil.which`` is patched through
``_resolve_cmake``.

CI collects this file from ``.github/workflows/unit_tests.yml`` because
``pyproject.toml`` lists ``build_tools/packaging/linux/tests``. Run locally
from the repository root::

    python3 -m unittest \\
        build_tools.packaging.linux.tests.downstream_cmake_smoke_test -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Resolve the packaging module from linux/ (same pattern as the other tests
# in this directory; linux/ is not an installed package).
_LINUX_DIR = Path(__file__).resolve().parent.parent
_path_str = os.fspath(_LINUX_DIR)
if _path_str not in sys.path:
    sys.path.insert(0, _path_str)

import downstream_cmake_smoke as smoke  # noqa: E402


class DownstreamCmakeSmokeTest(unittest.TestCase):
    """Command construction and fail-closed configure results."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="smoke-ut-")
        self.root = Path(self._temporary.name)
        self.prefix = self.root / "opt" / "rocm" / "core-test"
        self.prefix.mkdir(parents=True)
        self.work_dir = self.root / "work"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _fake_success(self, command: list[str]) -> smoke.ConfigureResult:
        self.command = command
        return smoke.ConfigureResult(
            returncode=0,
            stdout="-- hip_VERSION=1.0\n-- hipblaslt_VERSION=1.0\n",
            stderr="",
        )

    def test_generated_project_finds_hip_and_hipblaslt(self) -> None:
        """The generated project is configure-only and requires hip and hipblaslt."""
        contents = smoke.render_cmake_lists()
        self.assertIn("project(install_smoke_test LANGUAGES NONE)", contents)
        self.assertIn("find_package(hip CONFIG REQUIRED)", contents)
        self.assertIn("find_package(hipblaslt CONFIG REQUIRED)", contents)

    def test_configure_argv_uses_resolved_prefix_and_hip_platform(self) -> None:
        """Configure follows a symlink prefix and sets HIP_PLATFORM=amd."""
        link = self.root / "core-link"
        link.symlink_to(self.prefix, target_is_directory=True)

        with patch.object(smoke, "_resolve_cmake", return_value="/usr/bin/cmake"):
            return_code = smoke.run_smoke(
                link,
                work_dir=self.work_dir,
                run_configure=self._fake_success,
            )

        self.assertEqual(return_code, 0)
        self.assertIn(f"-DCMAKE_PREFIX_PATH={self.prefix.resolve()}", self.command)
        self.assertIn(f"-DHIP_PLATFORM={smoke.HIP_PLATFORM}", self.command)
        source_dir = Path(self.command[self.command.index("-S") + 1])
        self.assertFalse(smoke.path_is_inside(source_dir, self.prefix))
        self.assertTrue((source_dir / "CMakeLists.txt").is_file())

    def test_nonzero_configure_fails(self) -> None:
        """A non-zero cmake status fails the helper."""

        def fail_configure(_command: list[str]) -> smoke.ConfigureResult:
            return smoke.ConfigureResult(
                returncode=1,
                stdout="",
                stderr=(
                    "Could not find a package configuration file "
                    "provided by hipblaslt"
                ),
            )

        with patch.object(smoke, "_resolve_cmake", return_value="/usr/bin/cmake"):
            return_code = smoke.run_smoke(
                self.prefix,
                work_dir=self.work_dir,
                run_configure=fail_configure,
            )

        self.assertEqual(return_code, 1)

    def test_missing_cmake_fails(self) -> None:
        """A missing cmake executable fails instead of skipping."""
        with patch.object(smoke, "_resolve_cmake", return_value=None):
            return_code = smoke.run_smoke(self.prefix, work_dir=self.work_dir)

        self.assertEqual(return_code, 1)

    def test_project_inside_prefix_fails(self) -> None:
        """The smoke project is not written inside the install prefix."""
        inside = self.prefix / "smoke-work"
        called = False

        def configure(_command: list[str]) -> smoke.ConfigureResult:
            nonlocal called
            called = True
            return smoke.ConfigureResult(returncode=0, stdout="", stderr="")

        with patch.object(smoke, "_resolve_cmake", return_value="/usr/bin/cmake"):
            return_code = smoke.run_smoke(
                self.prefix,
                work_dir=inside,
                run_configure=configure,
            )

        self.assertEqual(return_code, 1)
        self.assertFalse(called)

    def test_missing_prefix_fails(self) -> None:
        """A prefix that is not a directory fails before cmake is invoked."""
        return_code = smoke.run_smoke(self.root / "missing-prefix")
        self.assertEqual(return_code, 1)

    def test_default_project_directory_is_removed(self) -> None:
        """The default temporary project is removed after configure."""
        source_dir: Path | None = None

        def configure(command: list[str]) -> smoke.ConfigureResult:
            nonlocal source_dir
            source_dir = Path(command[command.index("-S") + 1])
            cmake_lists = source_dir / "CMakeLists.txt"
            self.assertTrue(cmake_lists.is_file())
            self.assertGreater(cmake_lists.stat().st_size, 0)
            self.assertFalse(smoke.path_is_inside(source_dir, self.prefix))
            return self._fake_success(command)

        with patch.object(smoke, "_resolve_cmake", return_value="/usr/bin/cmake"):
            return_code = smoke.run_smoke(self.prefix, run_configure=configure)

        self.assertEqual(return_code, 0)
        self.assertIsNotNone(source_dir)
        assert source_dir is not None
        self.assertFalse(source_dir.exists())


if __name__ == "__main__":
    unittest.main()
