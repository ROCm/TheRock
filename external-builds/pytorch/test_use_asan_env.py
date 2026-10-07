# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_prod_wheels import _setup_common_build_env


def _plant_asan_runtime(root: Path) -> Path:
    runtime = (
        root
        / "lib"
        / "llvm"
        / "lib"
        / "clang"
        / "24"
        / "lib"
        / "x86_64-unknown-linux-gnu"
        / "libclang_rt.asan.so"
    )
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"")
    return runtime


class UseAsanEnvTest(unittest.TestCase):
    def test_use_asan_preloads_the_sdk_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _plant_asan_runtime(root)
            with mock.patch.dict(os.environ, {"USE_ASAN": "1"}, clear=False):
                os.environ.pop("LD_PRELOAD", None)
                os.environ.pop("ASAN_OPTIONS", None)
                env = _setup_common_build_env(root, root, root, "gfx1100", None, False)
                self.assertEqual(os.environ["LD_PRELOAD"], str(runtime))
                self.assertEqual(os.environ["ASAN_OPTIONS"], "detect_leaks=0")
        self.assertEqual(env["USE_ASAN"], "1")
        self.assertEqual(env["LD_PRELOAD"], str(runtime))
        self.assertEqual(env["ASAN_OPTIONS"], "detect_leaks=0")

    def test_use_asan_keeps_existing_preload_and_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _plant_asan_runtime(root)
            with mock.patch.dict(
                os.environ,
                {"USE_ASAN": "1", "LD_PRELOAD": "/tmp/other.so", "ASAN_OPTIONS": "halt_on_error=0"},
            ):
                env = _setup_common_build_env(root, root, root, "gfx1100", None, False)
        self.assertEqual(env["LD_PRELOAD"], f"{runtime}{os.pathsep}/tmp/other.so")
        self.assertEqual(env["ASAN_OPTIONS"], "halt_on_error=0:detect_leaks=0")

    def test_use_asan_requires_the_sdk_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.dict(os.environ, {"USE_ASAN": "1"}):
                with self.assertRaises(RuntimeError):
                    _setup_common_build_env(root, root, root, "gfx1100", None, False)

    def test_use_asan_is_absent_unless_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.dict(os.environ):
                os.environ.pop("USE_ASAN", None)
                os.environ.pop("LD_PRELOAD", None)
                build_env = _setup_common_build_env(
                    root, root, root, "gfx1100", None, False
                )
        self.assertNotIn("USE_ASAN", build_env)
        self.assertNotIn("LD_PRELOAD", build_env)


if __name__ == "__main__":
    unittest.main()
