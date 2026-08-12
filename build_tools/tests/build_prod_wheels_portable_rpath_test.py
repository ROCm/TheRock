# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for portable RPATH handling in the PyTorch wheel builder."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(
    0, os.fspath(Path(__file__).parent.parent.parent / "external-builds" / "pytorch")
)

import build_prod_wheels as bpw


class PortableRpathEnvironmentTest(unittest.TestCase):
    def test_appends_to_caller_cmake_args(self):
        caller_args = "-DFOO=ON -DQUOTED='value with spaces'"
        with mock.patch.dict(os.environ, {"CMAKE_ARGS": caller_args}):
            env = {}
            bpw.enable_pytorch_portable_rpath(env)

        self.assertEqual(
            env["CMAKE_ARGS"],
            caller_args + " -DTHEROCK_PYTORCH_PORTABLE_RPATH=ON",
        )

    def test_explicit_build_env_takes_precedence(self):
        env = {"CMAKE_ARGS": "-DBAR=OFF"}
        with mock.patch.dict(os.environ, {"CMAKE_ARGS": "-DFOO=ON"}):
            bpw.enable_pytorch_portable_rpath(env)

        self.assertEqual(
            env["CMAKE_ARGS"],
            "-DBAR=OFF -DTHEROCK_PYTORCH_PORTABLE_RPATH=ON",
        )


class PortableRpathValidationTest(unittest.TestCase):
    @staticmethod
    def _write_wheel(root: Path) -> Path:
        wheel_path = root / "torch-test.whl"
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr("torch/lib/libtorch_cpu.so", b"\x7fELFtest")
            wheel.writestr("torch/lib/README", b"not an ELF")
        return wheel_path

    def test_accepts_origin_relative_runpath(self):
        with tempfile.TemporaryDirectory() as td:
            wheel_path = self._write_wheel(Path(td))
            dynamic_section = (
                "0x000000000000001d (RUNPATH) Library runpath: "
                "[$ORIGIN:$ORIGIN/../../_rocm_sdk_core/lib]"
            )
            with mock.patch.object(
                bpw.shutil, "which", return_value="/usr/bin/readelf"
            ), mock.patch.object(
                bpw.subprocess, "check_output", return_value=dynamic_section
            ):
                bpw.validate_pytorch_wheel_runpaths(wheel_path)

    def test_rejects_absolute_build_runpath(self):
        with tempfile.TemporaryDirectory() as td:
            wheel_path = self._write_wheel(Path(td))
            dynamic_section = (
                "0x000000000000001d (RUNPATH) Library runpath: "
                "[/therock/output/build-venv/lib:$ORIGIN]"
            )
            with mock.patch.object(
                bpw.shutil, "which", return_value="/usr/bin/readelf"
            ), mock.patch.object(
                bpw.subprocess, "check_output", return_value=dynamic_section
            ):
                with self.assertRaisesRegex(RuntimeError, "/therock/output"):
                    bpw.validate_pytorch_wheel_runpaths(wheel_path)

    def test_parser_checks_rpath_and_runpath(self):
        dynamic_section = "\n".join(
            [
                "0x000000000000000f (RPATH) Library rpath: [$ORIGIN/one]",
                "0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/two]",
            ]
        )
        self.assertEqual(
            bpw._readelf_search_paths(dynamic_section),
            ["$ORIGIN/one", "$ORIGIN/two"],
        )


if __name__ == "__main__":
    unittest.main()
