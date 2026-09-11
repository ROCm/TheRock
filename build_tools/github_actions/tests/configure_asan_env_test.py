#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_asan_env.py"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_asan_env import (
    STATIC_ASAN_ENV,
    _asan_runtime_library_name,
    main,
    resolve_asan_env,
)

# The runtime is located by executing the artifact tree's clang, which the tests
# stub with a shell script. Windows has no equivalent shebang mechanism.
requires_posix = unittest.skipIf(
    os.name != "posix", "stub executables require a POSIX shell"
)


def _make_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_artifacts(root: Path, *, clang=True, symbolizer=True, runtime=True) -> Path:
    """Builds a fake artifact tree; returns the artifacts dir."""
    artifacts = root / "build"
    runtime_path = (
        artifacts
        / "lib"
        / "llvm"
        / "lib"
        / "clang"
        / "24"
        / "libclang_rt.asan-x86_64.so"
    )
    if runtime:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text("")
    if clang:
        _make_executable(
            artifacts / "llvm" / "bin" / "clang",
            f'#!/bin/sh\necho "{runtime_path}"\n',
        )
    if symbolizer:
        _make_executable(artifacts / "llvm" / "bin" / "llvm-symbolizer", "#!/bin/sh\n")
    return artifacts


class TestResolveAsanEnv(unittest.TestCase):
    def test_runtime_library_name_normalizes_host_architecture(self):
        for machine, expected_arch in (
            ("x86_64", "x86_64"),
            ("AMD64", "x86_64"),
            ("aarch64", "aarch64"),
            ("ARM64", "aarch64"),
        ):
            with self.subTest(machine=machine), patch(
                "configure_asan_env.platform.machine", return_value=machine
            ):
                self.assertEqual(
                    _asan_runtime_library_name(),
                    f"libclang_rt.asan-{expected_arch}.so",
                )

    @requires_posix
    def test_resolves_runtime_and_symbolizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, warnings = resolve_asan_env(artifacts)

            self.assertEqual(warnings, [])
            self.assertTrue(
                env["ASAN_RUNTIME_PATH"].endswith("libclang_rt.asan-x86_64.so")
            )
            self.assertTrue(env["ASAN_SYMBOLIZER_PATH"].endswith("llvm-symbolizer"))

    def test_static_values_are_always_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = resolve_asan_env(Path(tmp) / "missing")
            for key, value in STATIC_ASAN_ENV.items():
                self.assertEqual(env[key], value)

    def test_link_order_check_is_disabled(self):
        """Uninstrumented tools (amdclang++) abort without this."""
        self.assertIn("verify_asan_link_order=0", STATIC_ASAN_ENV["ASAN_OPTIONS"])

    @requires_posix
    def test_symbolizer_path_is_absolute_for_a_relative_artifacts_dir(self):
        """hip_check runs from build/bin, so a relative path resolves to nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_artifacts(Path(tmp))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                env, _ = resolve_asan_env(Path("./build"))
            finally:
                os.chdir(cwd)

            self.assertTrue(Path(env["ASAN_SYMBOLIZER_PATH"]).is_absolute())
            self.assertTrue(Path(env["ASAN_RUNTIME_PATH"]).is_absolute())

    def test_missing_clang_warns_without_exporting_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), clang=False)
            env, warnings = resolve_asan_env(artifacts)

            self.assertNotIn("ASAN_RUNTIME_PATH", env)
            self.assertTrue(any("clang not found" in w for w in warnings))

    @requires_posix
    def test_missing_runtime_file_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), runtime=False)
            env, warnings = resolve_asan_env(artifacts)

            self.assertNotIn("ASAN_RUNTIME_PATH", env)
            self.assertTrue(any("ASAN runtime not found" in w for w in warnings))

    @requires_posix
    def test_missing_symbolizer_warns_but_keeps_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), symbolizer=False)
            env, warnings = resolve_asan_env(artifacts)

            self.assertIn("ASAN_RUNTIME_PATH", env)
            self.assertNotIn("ASAN_SYMBOLIZER_PATH", env)
            self.assertTrue(any("unsymbolized" in w for w in warnings))


class TestMain(unittest.TestCase):
    @requires_posix
    def test_writes_key_value_lines_to_github_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env_file = Path(tmp) / "github_env"
            env_file.touch()

            previous = os.environ.get("GITHUB_ENV")
            os.environ["GITHUB_ENV"] = str(env_file)
            try:
                self.assertEqual(main(["--artifacts-dir", str(artifacts)]), 0)
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
            self.assertIn("ASAN_SYMBOLIZER_PATH", written)


if __name__ == "__main__":
    unittest.main()
