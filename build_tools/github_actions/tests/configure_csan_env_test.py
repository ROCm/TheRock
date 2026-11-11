#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise CSAN artifact discovery and exported test environments."""

from contextlib import chdir, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configure_csan_env import (
    CsanEnvironmentError,
    REPORT_PRELOAD_NAME,
    build_report_preload,
    main,
    resolve_csan_env,
)

HOST_C_COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def make_artifacts(
    root: Path,
    *,
    runtime_name: str = "libclang_rt.csan.so",
    runtime: bool = True,
    hsa_runtime: bool = True,
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
    if hsa_runtime:
        hsa_path = artifacts / "lib/libhsa-runtime64.so.1"
        hsa_path.parent.mkdir(parents=True, exist_ok=True)
        hsa_path.write_text("test HSA runtime", encoding="utf-8")
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
    lines.extend(
        (
            'case "$1" in -print-*) printf "%s\\n" "${1#*=}"; exit 0;; esac',
            f"exec {shlex.quote(HOST_C_COMPILER or 'cc')} \"$@\"",
        )
    )
    compiler.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    return artifacts, runtime_path, symbolizer_path


# CSAN runs only on Linux x86-64. These tests deliberately exercise real POSIX
# compiler-query subprocesses, as the production helper does on that platform.
@unittest.skipUnless(
    os.name == "posix" and HOST_C_COMPILER,
    "CSAN environment integration requires POSIX and a C compiler",
)
class ConfigureCsanEnvTest(unittest.TestCase):
    def test_artifact_paths_and_existing_environment_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, symbolizer = make_artifacts(root)
            original = {
                "LD_PRELOAD": "/existing/first.so /existing/second.so",
                "LD_LIBRARY_PATH": "/existing/lib",
                "CSAN_OPTIONS": 'skip_watch=37:strip_path_prefix="old source/":halt_on_error=0:color=always:log_path=/wrong',
                "ASAN_OPTIONS": "detect_leaks=0",
                "HSA_XNACK": "1",
            }
            before = original.copy()
            reports = root / "reports"
            env, warnings = resolve_csan_env(artifacts, reports, environ=original)
            self.assertFalse(warnings)
            self.assertEqual(original, before)
            self.assertEqual(env["CSAN_RUNTIME_PATH"], str(runtime.resolve()))
            hsa = artifacts / "lib/libhsa-runtime64.so.1"
            self.assertEqual(env["CSAN_HSA_RUNTIME_PATH"], str(hsa))
            self.assertEqual(env["CSAN_REPORT_PREFIX"], str(reports / "csan"))
            self.assertEqual(env["CSAN_SYMBOLIZER_PATH"], str(symbolizer.resolve()))
            self.assertEqual(
                env["LD_PRELOAD"],
                f"{runtime.resolve()}:{hsa}:{reports / REPORT_PRELOAD_NAME}:"
                + before["LD_PRELOAD"],
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
            self.assertEqual(
                env["LD_PRELOAD"],
                f"{runtime}:{artifacts / 'lib/libhsa-runtime64.so.1'}:"
                f"{root / 'reports' / REPORT_PRELOAD_NAME}",
            )

    def test_preload_paths_reject_loader_delimiters(self):
        for unsafe_name in ("path with spaces", "path:with:colons"):
            with self.subTest(path=unsafe_name):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    artifacts, _, _ = make_artifacts(root)
                    with self.assertRaisesRegex(
                        CsanEnvironmentError, "report initializer preload path"
                    ):
                        resolve_csan_env(artifacts, root / unsafe_name, environ={})

                    unsafe_artifacts, _, _ = make_artifacts(root / unsafe_name)
                    with self.assertRaisesRegex(
                        CsanEnvironmentError, "CSAN runtime preload path"
                    ):
                        resolve_csan_env(unsafe_artifacts, root / "reports", environ={})

                    hsa = artifacts / "lib/libhsa-runtime64.so.1"
                    unsafe_hsa = root / (unsafe_name + ".so")
                    hsa.rename(unsafe_hsa)
                    hsa.symlink_to(unsafe_hsa)
                    with self.assertRaisesRegex(
                        CsanEnvironmentError, "HSA runtime preload path"
                    ):
                        resolve_csan_env(artifacts, root / "reports", environ={})

    def test_missing_hsa_runtime_fails_instead_of_using_system_hsa(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root, hsa_runtime=False)
            with self.assertRaisesRegex(CsanEnvironmentError, "HSA runtime not found"):
                resolve_csan_env(artifacts, root / "reports", environ={})

    def test_unversioned_hsa_runtime_is_resolved_after_csan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, _ = make_artifacts(root)
            versioned = artifacts / "lib/libhsa-runtime64.so.1"
            unversioned = artifacts / "lib/libhsa-runtime64.so"
            versioned.rename(unversioned)
            env, _ = resolve_csan_env(artifacts, root / "reports", environ={})
            self.assertEqual(
                env["LD_PRELOAD"],
                f"{runtime}:{unversioned}:{root / 'reports' / REPORT_PRELOAD_NAME}",
            )

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
            self.assertIn(
                f"LD_PRELOAD={runtime}:{artifacts / 'lib/libhsa-runtime64.so.1'}:"
                f"{reports / REPORT_PRELOAD_NAME}\n",
                written,
            )
            self.assertIn('log_path="' + str(reports / "csan") + '"', written)
            self.assertNotIn("ASAN_OPTIONS=", written)
            self.assertNotIn("HSA_XNACK=", written)
            self.assertTrue((reports / REPORT_PRELOAD_NAME).is_file())

    def test_native_preload_constructor_sets_report_path_before_python_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, runtime, _ = make_artifacts(root)
            fake_runtime_source = root / "fake-runtime.c"
            fake_runtime_source.write_text(
                "#include <stdio.h>\n#include <string.h>\n#include <unistd.h>\n"
                "static char prefix[4096];\n"
                "void __sanitizer_set_report_path(const char *path) {\n"
                '  snprintf(prefix, sizeof(prefix), "%s", path);\n}\n'
                "const char *__sanitizer_get_report_path(void) { return prefix; }\n"
                "void emit_fake_race(void) {\n"
                "  char path[4200];\n"
                '  snprintf(path, sizeof(path), "%s.%ld", prefix, (long)getpid());\n'
                '  FILE *report = fopen(path, "w");\n'
                "  if (report) {\n"
                '    fputs("WARNING: ConcurrencySanitizer: intra-wave data race\\n", report);\n'
                "    fclose(report);\n  }\n}\n",
                encoding="utf-8",
            )
            hsa_source = root / "fake-hsa.c"
            hsa_source.write_text(
                "int hsa_stub(void) { return 0; }\n", encoding="utf-8"
            )
            for source, library in (
                (fake_runtime_source, runtime),
                (hsa_source, artifacts / "lib/libhsa-runtime64.so.1"),
            ):
                subprocess.run(
                    [
                        HOST_C_COMPILER,
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        str(library),
                    ],
                    check=True,
                    capture_output=True,
                )
            reports = root / "reports"
            build_report_preload(artifacts, reports)
            env, _ = resolve_csan_env(artifacts, reports, environ={})
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes, os; rt = ctypes.CDLL(None); "
                    "rt.__sanitizer_get_report_path.restype = ctypes.c_char_p; "
                    "assert rt.__sanitizer_get_report_path().decode() == os.environ['CSAN_REPORT_PREFIX']; "
                    "rt.emit_fake_race()",
                ],
                env={**os.environ, **env},
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            files = list(reports.glob("csan.*"))
            self.assertEqual(len(files), 1)
            self.assertIn("intra-wave data race", files[0].read_text(encoding="utf-8"))

    def test_failed_shim_compile_does_not_export_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _, _ = make_artifacts(root)
            invalid_source = root / "invalid.c"
            invalid_source.write_text("this is not valid C;\n", encoding="utf-8")
            env_file = root / "github-env"
            with patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}, clear=True):
                with patch("configure_csan_env.REPORT_PRELOAD_SOURCE", invalid_source):
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
            self.assertFalse((root / "reports" / REPORT_PRELOAD_NAME).exists())

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
