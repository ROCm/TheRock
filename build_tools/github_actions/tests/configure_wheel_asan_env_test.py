#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_wheel_asan_env.py"""

import os
import platform
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_asan_env import STATIC_ASAN_ENV
from configure_wheel_asan_env import (
    find_asan_runtime,
    find_symbolizer,
    main,
    resolve_wheel_asan_env,
)

requires_posix = unittest.skipIf(
    os.name != "posix", "stub executables require a POSIX shell"
)

ARCH_RUNTIME_NAME = f"libclang_rt.asan-{platform.machine()}.so"


def _make_package(
    root: Path,
    *,
    runtime_names: tuple[str, ...] = ("libclang_rt.asan.so",),
    symbolizer: bool = True,
) -> Path:
    """Builds a fake installed core package tree; returns its root."""
    package = root / "_rocm_sdk_core"
    lib_dir = package / "lib" / "llvm" / "lib" / "clang" / "24" / "lib" / "linux"
    lib_dir.mkdir(parents=True, exist_ok=True)
    for name in runtime_names:
        (lib_dir / name).write_text("")
    if symbolizer:
        bin_dir = package / "lib" / "llvm" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        path = bin_dir / "llvm-symbolizer"
        path.write_text("#!/bin/sh\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return package


class TestFindAsanRuntime(unittest.TestCase):
    def test_resolves_the_per_target_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp))
            runtime, warning = find_asan_runtime(package)

            self.assertIsNone(warning)
            self.assertEqual(runtime.name, "libclang_rt.asan.so")
            self.assertTrue(runtime.is_absolute())

    def test_resolves_the_arch_suffixed_runtime(self):
        """#8077 reintroduced the layout that names the runtime by arch."""
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp), runtime_names=(ARCH_RUNTIME_NAME,))
            runtime, warning = find_asan_runtime(package)

            self.assertIsNone(warning)
            self.assertEqual(runtime.name, ARCH_RUNTIME_NAME)

    def test_prefers_the_per_target_name_when_both_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(
                Path(tmp), runtime_names=("libclang_rt.asan.so", ARCH_RUNTIME_NAME)
            )
            runtime, _ = find_asan_runtime(package)

            self.assertEqual(runtime.name, "libclang_rt.asan.so")

    def test_non_instrumented_wheels_warn_without_a_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp), runtime_names=())
            runtime, warning = find_asan_runtime(package)

            self.assertIsNone(runtime)
            self.assertIn("not ASAN-instrumented", warning)


class TestResolveWheelAsanEnv(unittest.TestCase):
    def test_static_values_are_always_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = resolve_wheel_asan_env(Path(tmp) / "missing")
            for key, value in STATIC_ASAN_ENV.items():
                self.assertEqual(env[key], value)

    @requires_posix
    def test_exports_runtime_and_symbolizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp))
            env, warnings = resolve_wheel_asan_env(package)

            self.assertEqual(warnings, [])
            self.assertTrue(env["ASAN_RUNTIME_PATH"].endswith("libclang_rt.asan.so"))
            self.assertTrue(env["ASAN_SYMBOLIZER_PATH"].endswith("llvm-symbolizer"))

    def test_missing_symbolizer_warns_but_keeps_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp), symbolizer=False)
            env, warnings = resolve_wheel_asan_env(package)

            self.assertIn("ASAN_RUNTIME_PATH", env)
            self.assertNotIn("ASAN_SYMBOLIZER_PATH", env)
            self.assertTrue(any("unsymbolized" in w for w in warnings))


class TestFindSymbolizer(unittest.TestCase):
    def test_ignores_a_non_executable_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp), symbolizer=False)
            path = package / "lib" / "llvm" / "bin" / "llvm-symbolizer"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            path.chmod(0o644)

            self.assertIsNone(find_symbolizer(package))


class TestMain(unittest.TestCase):
    def test_writes_key_value_lines_to_github_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _make_package(Path(tmp))
            env_file = Path(tmp) / "github_env"
            env_file.touch()

            previous = os.environ.get("GITHUB_ENV")
            os.environ["GITHUB_ENV"] = str(env_file)
            try:
                self.assertEqual(main(["--package-root", str(package)]), 0)
            finally:
                if previous is None:
                    del os.environ["GITHUB_ENV"]
                else:
                    os.environ["GITHUB_ENV"] = previous

            written = dict(
                line.split("=", 1) for line in env_file.read_text().splitlines() if line
            )
            self.assertEqual(written["ASAN_OPTIONS"], STATIC_ASAN_ENV["ASAN_OPTIONS"])
            self.assertIn("ASAN_RUNTIME_PATH", written)


if __name__ == "__main__":
    unittest.main()
