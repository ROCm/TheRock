#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise CSAN artifact discovery and exported test environments."""

from contextlib import chdir, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configure_csan_env import CsanEnvironmentError, main, resolve_csan_env


def make_artifacts(
    root: Path,
    *,
    runtime_name: str = "libclang_rt.csan.so",
    runtime: bool = True,
    symbolizer: bool = True,
    compiler_layout: str = "llvm/bin",
) -> tuple[Path, Path, Path]:
    artifacts = root / "artifacts"
    runtime_path = (
        artifacts / "lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu" / runtime_name
    )
    symbolizer_path = artifacts / compiler_layout / "llvm-symbolizer"
    if runtime:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text("test runtime", encoding="utf-8")
    if symbolizer:
        symbolizer_path.parent.mkdir(parents=True, exist_ok=True)
        symbolizer_path.write_text("test symbolizer", encoding="utf-8")
    compiler = artifacts / compiler_layout / "clang"
    compiler.parent.mkdir(parents=True, exist_ok=True)
    # Run real subprocesses against an artifact-owned compiler stub. Echoing
    # unknown names reproduces clang's missing-file query behavior.
    lines = ["#!/bin/sh"]
    if runtime:
        lines.append(
            f'if [ "$1" = "-print-file-name={runtime_name}" ]; then '
            f"printf '%s\\n' {shlex.quote(str(runtime_path))}; exit 0; fi"
        )
    if symbolizer:
        lines.append(
            'if [ "$1" = "-print-prog-name=llvm-symbolizer" ]; then '
            f"printf '%s\\n' {shlex.quote(str(symbolizer_path))}; exit 0; fi"
        )
    lines.append('printf "%s\\n" "${1#*=}"')
    compiler.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    return artifacts, runtime_path, symbolizer_path


# CSAN runs only on Linux x86-64. These tests deliberately exercise real POSIX
# compiler-query subprocesses, as the production helper does on that platform.
@unittest.skipUnless(os.name == "posix", "CSAN compiler queries require POSIX")
class ConfigureCsanEnvTest(unittest.TestCase):
    def test_artifact_paths_and_existing_environment_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, symbolizer = make_artifacts(root)
            original = {
                "LD_PRELOAD": "/existing/first.so /existing/second.so",
                "LD_LIBRARY_PATH": "/existing/lib",
                "CSAN_OPTIONS": "skip_watch=37:halt_on_error=0:color=always:log_path=/wrong",
                "ASAN_OPTIONS": "detect_leaks=0",
                "HSA_XNACK": "1",
            }
            before = original.copy()
            reports = root / "reports with spaces"
            env, warnings = resolve_csan_env(artifacts, reports, environ=original)
            self.assertFalse(warnings)
            self.assertEqual(original, before)
            self.assertEqual(env["CSAN_RUNTIME_PATH"], str(runtime.resolve()))
            self.assertEqual(env["CSAN_SYMBOLIZER_PATH"], str(symbolizer.resolve()))
            self.assertEqual(
                env["LD_PRELOAD"], str(runtime.resolve()) + ":" + before["LD_PRELOAD"]
            )
            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(":"),
                [
                    str(artifacts / "lib"),
                    str(artifacts / "lib/rocm_sysdeps/lib"),
                    str(runtime.parent),
                    "/existing/lib",
                ],
            )
            self.assertEqual(
                env["CSAN_OPTIONS"],
                before["CSAN_OPTIONS"]
                + f':halt_on_error=1:color=never:log_path="{reports.resolve() / "csan"}"',
            )
            self.assertNotIn("ASAN_OPTIONS", env)
            self.assertNotIn("HSA_XNACK", env)

    def test_legacy_runtime_and_alternate_artifact_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, _ = make_artifacts(
                root,
                runtime_name="libclang_rt.csan-x86_64.so",
                compiler_layout="lib/llvm/bin",
            )
            env, _ = resolve_csan_env(artifacts, root / "reports", environ={})
            self.assertEqual(env["LD_PRELOAD"], str(runtime))

    def test_missing_artifact_compiler_does_not_use_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root)
            (artifacts / "llvm/bin/clang").unlink()
            other_artifacts, _, _ = make_artifacts(root / "system")
            compiler_path = str(other_artifacts / "llvm/bin")
            with patch.dict(os.environ, {"PATH": compiler_path}):
                with self.assertRaisesRegex(
                    CsanEnvironmentError, "No artifact compiler"
                ):
                    resolve_csan_env(
                        artifacts, root / "reports", environ={"PATH": compiler_path}
                    )

    def test_missing_runtime_rejects_clang_echo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root, runtime=False)
            (root / "libclang_rt.csan.so").write_text(
                "unrelated local file", encoding="utf-8"
            )
            with chdir(root):
                with self.assertRaisesRegex(
                    CsanEnvironmentError, "CSAN runtime not found"
                ):
                    resolve_csan_env(artifacts, root / "reports", environ={})

    def test_missing_symbolizer_warns_without_using_system_symbolizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root, symbolizer=False)
            env, warnings = resolve_csan_env(artifacts, root / "reports", environ={})
            self.assertNotIn("CSAN_SYMBOLIZER_PATH", env)
            self.assertEqual(len(warnings), 1)
            self.assertIn("unsymbolized", warnings[0])

    def test_query_failure_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root)
            compiler = artifacts / "llvm/bin/clang"
            compiler.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
            with self.assertRaisesRegex(CsanEnvironmentError, "artifact compiler"):
                resolve_csan_env(artifacts, root / "reports", environ={})

    def test_main_creates_reports_and_exports_without_process_pollution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, _ = make_artifacts(root)
            reports = root / "reports"
            env_file = root / "github-env"
            with patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}, clear=True):
                before = os.environ.copy()
                with redirect_stdout(io.StringIO()):
                    result = main(
                        [
                            "--artifacts-dir",
                            str(artifacts),
                            "--reports-dir",
                            str(reports),
                        ]
                    )
                self.assertEqual(os.environ, before)
            self.assertEqual(result, 0)
            self.assertTrue(reports.is_dir())
            written = env_file.read_text(encoding="utf-8")
            self.assertIn(f"LD_PRELOAD={runtime}\n", written)
            self.assertIn('log_path="' + str(reports / "csan") + '"', written)
            self.assertNotIn("ASAN_OPTIONS=", written)
            self.assertNotIn("HSA_XNACK=", written)

    def test_failure_does_not_write_partial_github_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root, runtime=False)
            env_file = root / "github-env"
            with patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}, clear=True):
                with redirect_stderr(io.StringIO()):
                    result = main(
                        [
                            "--artifacts-dir",
                            str(artifacts),
                            "--reports-dir",
                            str(root / "reports"),
                        ]
                    )
            self.assertEqual(result, 1)
            self.assertFalse(env_file.exists())


if __name__ == "__main__":
    unittest.main()
