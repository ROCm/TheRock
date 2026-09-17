#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_tsan_env.py."""

import os
import stat
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_tsan_env import STATIC_TSAN_OPTIONS, main, resolve_tsan_env

requires_posix = unittest.skipIf(
    os.name != "posix", "stub executables require a POSIX shell"
)


def _make_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_artifacts(root: Path, *, clang=True, symbolizer=True, runtime=True) -> Path:
    artifacts = root / "build"
    runtime_path = (
        artifacts
        / "lib"
        / "llvm"
        / "lib"
        / "clang"
        / "24"
        / "libclang_rt.tsan-x86_64.so"
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


class TestResolveTsanEnv(unittest.TestCase):
    @requires_posix
    def test_resolves_runtime_and_symbolizer_without_preload(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp))
            env, warnings = resolve_tsan_env(artifacts)

            self.assertEqual(warnings, [])
            self.assertTrue(env["TSAN_RUNTIME_PATH"].endswith(".so"))
            self.assertTrue(env["TSAN_SYMBOLIZER_PATH"].endswith("llvm-symbolizer"))
            self.assertNotIn("LD_PRELOAD", env)
            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(":")[0],
                str(Path(env["TSAN_RUNTIME_PATH"]).parent),
            )
            self.assertIn(
                str(artifacts / "lib" / "llvm" / "lib"),
                env["LD_LIBRARY_PATH"].split(":"),
            )

    def test_deterministic_failure_options_are_always_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = resolve_tsan_env(Path(tmp) / "missing")

        self.assertEqual(env["TSAN_OPTIONS"], STATIC_TSAN_OPTIONS)
        self.assertIn("halt_on_error=1", env["TSAN_OPTIONS"])
        self.assertIn("exitcode=86", env["TSAN_OPTIONS"])

    def test_missing_clang_warns_without_exporting_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), clang=False)
            env, warnings = resolve_tsan_env(artifacts)

        self.assertNotIn("TSAN_RUNTIME_PATH", env)
        self.assertTrue(any("clang not found" in warning for warning in warnings))

    @requires_posix
    def test_missing_runtime_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _make_artifacts(Path(tmp), runtime=False)
            env, warnings = resolve_tsan_env(artifacts)

        self.assertNotIn("TSAN_RUNTIME_PATH", env)
        self.assertTrue(any("runtime was not found" in warning for warning in warnings))


class TestMain(unittest.TestCase):
    def test_missing_runtime_fails_closed(self):
        with mock.patch(
            "configure_tsan_env.resolve_tsan_env",
            return_value=(
                {"TSAN_OPTIONS": STATIC_TSAN_OPTIONS, "LD_LIBRARY_PATH": "/rocm/lib"},
                ["runtime missing"],
            ),
        ):
            self.assertEqual(main(["--artifacts-dir", "/missing"]), 1)

    @requires_posix
    def test_writes_environment_file(self):
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
                line.split("=", 1)
                for line in env_file.read_text().splitlines()
                if line
            )
            self.assertIn("TSAN_RUNTIME_PATH", written)
            self.assertIn("TSAN_SYMBOLIZER_PATH", written)
            self.assertNotIn("LD_PRELOAD", written)


if __name__ == "__main__":
    unittest.main()
