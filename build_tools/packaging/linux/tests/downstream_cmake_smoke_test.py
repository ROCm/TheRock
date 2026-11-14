#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for downstream_cmake_smoke.py. No ROCm prefix or cmake required."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LINUX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LINUX_DIR))

import downstream_cmake_smoke as sdk  # noqa: E402


class CommandTests(unittest.TestCase):
    def test_command_uses_driver_and_prefix_env(self) -> None:
        """Pass the prefix as ``CMAKE_PREFIX_PATH`` and run the existing driver.

        ``THEROCK_ENABLE_*`` is left at the project default.
        """
        command = sdk.build_command(
            cmake="cmake",
            source_dir=Path("/src/examples/cpp-sdk-user"),
            binary_dir=Path("/tmp/cpp-sdk-user"),
            driver=Path("/src/examples/clean_configure_test_project.cmake"),
            prefix=Path("/opt/rocm/core"),
            generator="Unix Makefiles",
        )
        self.assertEqual(
            command[:5],
            ["cmake", "-E", "env", "CMAKE_PREFIX_PATH=/opt/rocm/core", "--"],
        )
        self.assertFalse(any(arg.startswith("-DTHEROCK_ENABLE_") for arg in command))
        self.assertEqual(
            command[-2:], ["-P", "/src/examples/clean_configure_test_project.cmake"]
        )

    def test_prefix_clang_is_the_host_compiler(self) -> None:
        """Use prefix Clang when ``--cxx-compiler`` is omitted.

        A file at ``<prefix>/lib/llvm/bin/clang++`` is the host compiler.
        An explicit compiler replaces that default and is returned unchanged.
        """
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            clangxx = prefix / "lib" / "llvm" / "bin" / "clang++"
            clangxx.parent.mkdir(parents=True)
            clangxx.write_text("")
            self.assertEqual(sdk.host_cxx_compiler(prefix), clangxx.as_posix())
            self.assertEqual(
                sdk.host_cxx_compiler(prefix, "/usr/bin/g++"), "/usr/bin/g++"
            )

    def test_missing_prefix_clang_leaves_the_compiler_unset(self) -> None:
        """Leave the host compiler unset when the prefix has no Clang.

        ``host_cxx_compiler`` returns None so CMake chooses the compiler.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sdk.host_cxx_compiler(Path(tmp)))

    def test_build_dir_inside_prefix_fails(self) -> None:
        """Reject a build directory inside the install prefix."""
        with (
            tempfile.TemporaryDirectory() as prefix_dir,
            tempfile.TemporaryDirectory() as other,
        ):
            prefix = Path(prefix_dir)
            other_path = Path(other)
            source = other_path / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(cpp_sdk_user)\n")
            driver = other_path / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            with self.assertRaisesRegex(ValueError, "inside the install prefix"):
                sdk.validate_layout(prefix, prefix / "build", source, driver)

    def test_main_invokes_cmake_without_enable_flags(self) -> None:
        """Run the driver and leave every component at its default of ON."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix"
            prefix.mkdir()
            source = root / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(cpp_sdk_user)\n")
            driver = root / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            cmake = root / "cmake"
            cmake.write_text("#!/bin/sh\n")
            os.chmod(cmake, 0o700)
            binary = root / "build"
            with patch.object(sdk.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                code = sdk.main(
                    [
                        "--prefix",
                        str(prefix),
                        "--source-dir",
                        str(source),
                        "--driver",
                        str(driver),
                        "--cmake",
                        str(cmake),
                        "--binary-dir",
                        str(binary),
                    ]
                )
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertFalse(any(arg.startswith("-DTHEROCK_ENABLE_") for arg in command))
        self.assertTrue(command[-1].endswith("clean_configure_test_project.cmake"))


class ArgumentAndRunTests(unittest.TestCase):
    def test_directory_is_rejected_as_cmake(self) -> None:
        """A directory is not an executable cmake on any platform."""
        with tempfile.TemporaryDirectory() as tmp:
            cmake = Path(tmp) / "cmake"
            cmake.mkdir()
            with patch.dict(os.environ, {"PATH": ""}):
                with self.assertRaisesRegex(ValueError, "cmake executable not found"):
                    sdk.resolve_cmake(str(cmake))

    def test_empty_prefix_is_rejected(self) -> None:
        """A blank ``--prefix`` must not resolve to the working directory."""
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                sdk.parse_args(["--prefix", ""])
        self.assertEqual(raised.exception.code, 2)

    def test_main_returns_one_when_cmake_cannot_start(self) -> None:
        """An OSError from subprocess is reported, not raised as a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix"
            prefix.mkdir()
            source = root / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(cpp_sdk_user)\n")
            driver = root / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            cmake = root / "cmake"
            cmake.write_text("#!/bin/sh\n")
            os.chmod(cmake, 0o700)
            argv = [
                "--prefix",
                str(prefix),
                "--source-dir",
                str(source),
                "--driver",
                str(driver),
                "--cmake",
                str(cmake),
                "--binary-dir",
                str(root / "build"),
            ]
            with (
                patch.object(
                    sdk.subprocess, "run", side_effect=PermissionError("denied")
                ),
                patch("sys.stderr", new=io.StringIO()) as err,
            ):
                code = sdk.main(argv)
        self.assertEqual(code, 1)
        self.assertIn("could not run cmake", err.getvalue())


if __name__ == "__main__":
    unittest.main()
