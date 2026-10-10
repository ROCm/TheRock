# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that Triton packaging uses the SDK's hostcall protocol."""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import (
    build_triton_linux,
    build_triton_windows,
    stage_triton_ockl,
)


class TritonOcklTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sdk_libdir = self.root / "sdk" / "lib" / "llvm" / "amdgcn" / "bitcode"
        self.sdk_libdir.mkdir(parents=True)
        self.sdk_ockl = self.sdk_libdir / "ockl.bc"
        self.sdk_ockl.write_bytes(b"SDK hostcall ABI")
        self.triton = self.root / "triton"
        source = self.triton / "third_party" / "amd" / "backend" / "lib"
        source.mkdir(parents=True)
        self.bundled_ockl = source / "ockl.bc"
        self.bundled_ockl.write_bytes(b"old hostcall ABI")
        self.ocml = source / "ocml.bc"
        self.ocml.write_bytes(b"Triton math library")
        (self.triton / "setup.py").touch()
        self.args = argparse.Namespace(
            version_suffix="",
            pip_cache_dir=None,
            clean=False,
            output_dir=self.root / "output",
        )
        self.env = {"HIP_DEVICE_LIB_PATH": str(self.sdk_libdir)}

    def assert_source_restored(self):
        self.assertEqual(self.bundled_ockl.read_bytes(), b"old hostcall ABI")
        self.assertEqual(self.ocml.read_bytes(), b"Triton math library")

    def test_incremental_package_replaces_old_hostcall_library(self):
        os.utime(self.sdk_ockl, (1, 1))
        destination = self.root / "build" / "ockl.bc"
        destination.parent.mkdir()
        destination.write_bytes(b"old hostcall ABI")
        os.utime(destination, (2, 2))
        with stage_triton_ockl(self.triton, self.env):
            # Exercise setuptools' timestamp-sensitive package-data copy.
            command = build_py(Distribution())
            command.force = False
            command.copy_file(str(self.bundled_ockl), str(destination))
            self.assertEqual(destination.read_bytes(), self.sdk_ockl.read_bytes())
        self.assert_source_restored()

    def test_missing_sdk_library_preserves_triton_library(self):
        self.sdk_ockl.unlink()
        with self.assertRaises(FileNotFoundError):
            with stage_triton_ockl(self.triton, self.env):
                self.fail("Missing SDK library should fail before staging")
        self.assert_source_restored()

    def test_missing_device_lib_path(self):
        with self.assertRaisesRegex(ValueError, "HIP_DEVICE_LIB_PATH"):
            with stage_triton_ockl(self.triton, {}):
                self.fail("Missing device library path should fail before staging")
        self.assert_source_restored()

    def run_build(
        self,
        *,
        windows=False,
        stale_payload=False,
        missing_payload=False,
        fail_build=False,
        fail_compile=False
    ):
        """Stub subprocesses, but exercise staging, wheel checking, and publication."""
        compile_checks = []

        def run_command(command, *, cwd, env=None):
            if command[1:] in (["setup.py", "bdist_wheel"], ["-m", "build", "--wheel"]):
                if fail_build:
                    raise subprocess.CalledProcessError(1, command)
                dist = Path(cwd) / "dist"
                dist.mkdir()
                payload = (
                    b"old hostcall ABI"
                    if stale_payload
                    else self.bundled_ockl.read_bytes()
                )
                with zipfile.ZipFile(
                    dist / "triton-3.8.0-py3-none-any.whl", "w"
                ) as wheel:
                    if not missing_payload:
                        wheel.writestr("triton/backends/amd/lib/ockl.bc", payload)
            elif command[-1] == "--compile-only":
                self.assert_source_restored()
                self.assertFalse(self.args.output_dir.exists())
                self.assertTrue(Path(env["TRITON_CACHE_DIR"]).is_dir())
                self.assertEqual(env["TRITON_INTERPRET"], "0")
                compile_checks.append(command)
                if fail_compile:
                    raise subprocess.CalledProcessError(1, command)

        with mock.patch(
            "build_prod_wheels.run_command", side_effect=run_command
        ), mock.patch(
            "build_prod_wheels.get_installed_package_version", return_value="3.8.0"
        ), mock.patch(
            "build_prod_wheels.download_llvm_for_triton_windows",
            return_value=self.root / "llvm",
        ):
            builder = build_triton_windows if windows else build_triton_linux
            result = builder(self.args, self.triton, self.env)
        self.assertEqual(result, "triton==3.8.0")
        self.assertEqual(len(compile_checks), 0 if windows else 1)

    def test_linux_builder_packages_sdk_library_and_restores_checkout(self):
        self.run_build()
        with zipfile.ZipFile(next(self.args.output_dir.glob("*.whl"))) as wheel:
            self.assertEqual(
                wheel.read("triton/backends/amd/lib/ockl.bc"),
                self.sdk_ockl.read_bytes(),
            )
        self.assert_source_restored()

    def test_windows_builder_packages_sdk_library_and_restores_checkout(self):
        self.run_build(windows=True)
        with zipfile.ZipFile(next(self.args.output_dir.glob("*.whl"))) as wheel:
            self.assertEqual(
                wheel.read("triton/backends/amd/lib/ockl.bc"),
                self.sdk_ockl.read_bytes(),
            )
        self.assert_source_restored()

    def test_incorrect_wheel_is_rejected_and_checkout_restored(self):
        for windows in (False, True):
            for missing_payload in (False, True):
                with self.subTest(windows=windows, missing_payload=missing_payload):
                    with self.assertRaisesRegex(
                        ValueError, "did not package the SDK's OCKL"
                    ):
                        self.run_build(
                            windows=windows,
                            stale_payload=True,
                            missing_payload=missing_payload,
                        )
                    self.assert_source_restored()
                    self.assertFalse(self.args.output_dir.exists())

    def test_failed_build_restores_checkout(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build(fail_build=True)
        self.assert_source_restored()
        self.assertFalse(self.args.output_dir.exists())

    def test_incompatible_bitcode_is_not_published(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build(fail_compile=True)
        self.assert_source_restored()
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
