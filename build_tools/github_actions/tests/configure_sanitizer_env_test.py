#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise sanitizer runtime discovery and environment export behavior."""

import contextlib
import io
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))
from configure_sanitizer_env import main, resolve_sanitizer_env


def executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)


def artifacts(
    root: Path, sanitizer: str, *, legacy: bool = False, native: bool = False
) -> Path:
    name = (
        f"libclang_rt.{sanitizer}"
        + (f"-{platform.machine()}" if legacy else "")
        + ".so"
    )
    runtime = root / "lib/llvm/lib/clang/24/lib" / name
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.touch()
    tool_dir = root / ("lib/llvm/bin" if native else "llvm/bin")
    executable(
        tool_dir / "clang",
        "#!/bin/sh\n"
        f'if [ "$1" = "-print-file-name={name}" ]; then\n'
        f"  printf '%s\\n' {shlex.quote(str(runtime))}\n"
        'else\n  printf "%s\\n" "${1#*=}"\nfi\n',
    )
    executable(tool_dir / "llvm-symbolizer", "#!/bin/sh\n")
    return runtime


@unittest.skipUnless(
    os.name == "posix", "artifact compiler fixtures use POSIX executables"
)
class SanitizerEnvironmentTest(unittest.TestCase):
    def test_discovers_both_runtime_layouts_and_preserves_library_paths(self):
        for variant in ("asan", "host-asan", "tsan"):
            for legacy in (False, True):
                for native in (False, True):
                    with self.subTest(variant=variant, legacy=legacy, native=native):
                        with tempfile.TemporaryDirectory() as tmp:
                            root = Path(tmp)
                            sanitizer = "tsan" if variant == "tsan" else "asan"
                            runtime = artifacts(
                                root, sanitizer, legacy=legacy, native=native
                            )
                            with mock.patch.dict(
                                os.environ, {"LD_LIBRARY_PATH": "/existing/lib"}
                            ):
                                env = resolve_sanitizer_env(root, variant)
                            self.assertEqual(
                                env[f"{sanitizer.upper()}_RUNTIME_PATH"], str(runtime)
                            )
                            self.assertTrue(
                                Path(
                                    env[f"{sanitizer.upper()}_SYMBOLIZER_PATH"]
                                ).is_absolute()
                            )
                            self.assertEqual(
                                env["LD_LIBRARY_PATH"].split(":")[0],
                                str(runtime.parent),
                            )
                            self.assertEqual(
                                env["LD_LIBRARY_PATH"].split(":")[-1], "/existing/lib"
                            )
                            self.assertNotIn("LD_PRELOAD", env)
                            if variant == "tsan":
                                self.assertNotIn("HSA_XNACK", env)
                                self.assertNotIn("ASAN_OPTIONS", env)
                            else:
                                self.assertNotIn("TSAN_OPTIONS", env)

    def test_missing_runtime_policy_and_no_partial_tsan_export(self):
        for variant, expected in (("asan", 0), ("host-asan", 0), ("tsan", 1)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "github_env"
                with mock.patch.dict(
                    os.environ, {"GITHUB_ENV": str(output)}
                ), self.assertLogs(level="WARNING"):
                    result = main(["--artifacts-dir", tmp, "--build-variant", variant])
                self.assertEqual(result, expected)
                if variant == "tsan":
                    self.assertFalse(output.exists())
                else:
                    self.assertNotIn("ASAN_RUNTIME_PATH", output.read_text())

    def test_compiler_failure_warns_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable(root / "llvm/bin/clang", "#!/bin/sh\nexit 2\n")
            with self.assertLogs(level="WARNING") as logs:
                env = resolve_sanitizer_env(root, "tsan")
            self.assertNotIn("TSAN_RUNTIME_PATH", env)
            self.assertIn("Could not query", logs.output[0])

    def test_missing_symbolizer_keeps_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = artifacts(root, "tsan")
            (root / "llvm/bin/llvm-symbolizer").unlink()
            with self.assertLogs(level="WARNING") as logs:
                env = resolve_sanitizer_env(root, "tsan")
            self.assertEqual(env["TSAN_RUNTIME_PATH"], str(runtime))
            self.assertNotIn("TSAN_SYMBOLIZER_PATH", env)
            self.assertTrue(any("unsymbolized" in line for line in logs.output))

    def test_exports_to_github_env(self):
        for variant in ("asan", "host-asan", "tsan"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                kind = "tsan" if variant == "tsan" else "asan"
                runtime = artifacts(root, kind)
                output = root / "github_env"
                with mock.patch.dict(os.environ, {"GITHUB_ENV": str(output)}):
                    self.assertEqual(
                        main(["--artifacts-dir", tmp, "--build-variant", variant]), 0
                    )
                exported = dict(
                    line.split("=", 1) for line in output.read_text().splitlines()
                )
                self.assertEqual(exported[f"{kind.upper()}_RUNTIME_PATH"], str(runtime))
                self.assertIn(f"{kind.upper()}_OPTIONS", exported)

    def test_shell_exports_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact tree"
            runtime = artifacts(root, "tsan")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(
                    main(
                        [
                            "--artifacts-dir",
                            str(root),
                            "--build-variant",
                            "tsan",
                            "--output-format",
                            "shell",
                        ]
                    ),
                    0,
                )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    captured.getvalue() + '\nprintf "%s" "$TSAN_RUNTIME_PATH"',
                ],
                cwd="/",
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout, str(runtime))

    def test_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            resolve_sanitizer_env(Path("/unused"), "release")


if __name__ == "__main__":
    unittest.main()
