#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for test_host_tsan_sanity.py."""

import os
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_host_tsan_sanity


@contextmanager
def _test_directory():
    root = Path(__file__).parent / f"_test_host_tsan_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class BuildEnvironmentTest(unittest.TestCase):
    def test_prepends_runtime_directory_and_sets_deterministic_options(self):
        runtime = Path("/rocm/llvm/lib/clang/24/libclang_rt.tsan.so")

        env = test_host_tsan_sanity.build_environment(
            runtime,
            {
                "LD_LIBRARY_PATH": "/existing",
                "LD_PRELOAD": "/forbidden/preload.so",
                "PRESERVED": "yes",
            },
        )

        self.assertEqual(env["LD_LIBRARY_PATH"], f"{runtime.parent}:/existing")
        self.assertEqual(env["TSAN_OPTIONS"], test_host_tsan_sanity.TSAN_OPTIONS)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["PRESERVED"], "yes")

    def test_preserves_configured_symbolizer(self):
        runtime = Path("/rocm/llvm/lib/clang/24/libclang_rt.tsan.so")
        env = test_host_tsan_sanity.build_environment(
            runtime,
            {"TSAN_SYMBOLIZER_PATH": "/rocm/llvm/bin/llvm-symbolizer"},
        )

        self.assertEqual(
            env["TSAN_OPTIONS"],
            test_host_tsan_sanity.TSAN_OPTIONS
            + ":external_symbolizer_path=/rocm/llvm/bin/llvm-symbolizer",
        )

    def test_omits_empty_trailing_library_path(self):
        runtime = Path("/rocm/libclang_rt.tsan.so")

        env = test_host_tsan_sanity.build_environment(runtime, {})

        self.assertEqual(env["LD_LIBRARY_PATH"], str(runtime.parent))


class CompileAndRunTest(unittest.TestCase):
    def test_compiles_with_tsan_flags_then_runs_the_binary(self):
        with _test_directory() as root:
            clang = Path("/rocm/llvm/bin/clang++")
            env = {"TSAN_OPTIONS": "exitcode=86"}
            executed = _completed(returncode=86, stderr="race")
            with mock.patch.object(
                test_host_tsan_sanity.subprocess,
                "run",
                side_effect=[_completed(), executed],
            ) as run:
                result = test_host_tsan_sanity.compile_and_run(
                    root, "canary", "int main() {}", clang, env
                )

            self.assertIs(result, executed)
            self.assertEqual((root / "canary.cpp").read_text(), "int main() {}")
            compile_command = run.call_args_list[0].args[0]
            self.assertEqual(compile_command[0], str(clang))
            self.assertIn("-fsanitize=thread", compile_command)
            self.assertIn("-shared-libsan", compile_command)
            self.assertEqual(
                compile_command[-4:],
                ["-pthread", str(root / "canary.cpp"), "-o", str(root / "canary")],
            )
            self.assertTrue(run.call_args_list[0].kwargs["check"])
            self.assertEqual(run.call_args_list[0].kwargs["env"], env)
            self.assertEqual(run.call_args_list[1].args[0], root / "canary")
            self.assertTrue(run.call_args_list[1].kwargs["capture_output"])


class RunCanariesTest(unittest.TestCase):
    def test_accepts_clean_run_and_deterministic_race_report(self):
        with _test_directory() as root, mock.patch.object(
            test_host_tsan_sanity.tempfile,
            "TemporaryDirectory",
            return_value=nullcontext(root),
        ), mock.patch.object(
            test_host_tsan_sanity,
            "compile_and_run",
            side_effect=[
                _completed(),
                _completed(
                    returncode=86,
                    stderr="WARNING: ThreadSanitizer: data race",
                ),
            ],
        ) as compile_and_run:
            test_host_tsan_sanity.run_canaries(Path("/clang++"), {})

        self.assertEqual(
            [call.args[1] for call in compile_and_run.call_args_list],
            ["clean", "race"],
        )

    def test_rejects_tsan_output_from_clean_canary(self):
        error = StringIO()
        with _test_directory() as root, mock.patch.object(
            test_host_tsan_sanity.tempfile,
            "TemporaryDirectory",
            return_value=nullcontext(root),
        ), mock.patch.object(
            test_host_tsan_sanity,
            "compile_and_run",
            return_value=_completed(stderr="ThreadSanitizer: unexpected"),
        ), redirect_stderr(
            error
        ), self.assertRaises(
            SystemExit
        ) as raised:
            test_host_tsan_sanity.run_canaries(Path("/clang++"), {})

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("clean TSAN canary failed", error.getvalue())

    def test_rejects_race_without_expected_exit_code_and_report(self):
        error = StringIO()
        with _test_directory() as root, mock.patch.object(
            test_host_tsan_sanity.tempfile,
            "TemporaryDirectory",
            return_value=nullcontext(root),
        ), mock.patch.object(
            test_host_tsan_sanity,
            "compile_and_run",
            side_effect=[_completed(), _completed(returncode=0)],
        ), redirect_stderr(
            error
        ), self.assertRaises(
            SystemExit
        ) as raised:
            test_host_tsan_sanity.run_canaries(Path("/clang++"), {})

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("racy TSAN canary did not produce", error.getvalue())


class MainTest(unittest.TestCase):
    def test_missing_artifact_clang_fails_closed(self):
        with _test_directory() as root:
            error = StringIO()
            environ = {"THEROCK_BIN_DIR": str(root / "bin")}
            with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                test_host_tsan_sanity.main(environ)

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("artifact clang++ not found", error.getvalue())

    def test_missing_runtime_fails_closed(self):
        with _test_directory() as root:
            clang = root / "llvm" / "bin" / "clang++"
            clang.parent.mkdir(parents=True)
            clang.touch()
            error = StringIO()
            environ = {
                "THEROCK_BIN_DIR": str(root / "bin"),
                "TSAN_RUNTIME_PATH": str(root / "missing-runtime.so"),
            }
            with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                test_host_tsan_sanity.main(environ)

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("TSAN runtime not found", error.getvalue())

    def test_runs_canaries_and_prints_success(self):
        with _test_directory() as root:
            clang = root / "llvm" / "bin" / "clang++"
            runtime = root / "llvm" / "lib" / "libclang_rt.tsan.so"
            clang.parent.mkdir(parents=True)
            runtime.parent.mkdir(parents=True)
            clang.touch()
            runtime.touch()
            environ = {
                "THEROCK_BIN_DIR": str(root / "bin"),
                "TSAN_RUNTIME_PATH": str(runtime),
                "LD_LIBRARY_PATH": "/existing",
            }
            output = StringIO()
            with mock.patch.object(
                test_host_tsan_sanity, "run_canaries"
            ) as run_canaries, mock.patch.object(
                test_host_tsan_sanity, "require_no_gpu_nodes"
            ) as no_gpu_nodes, redirect_stdout(output):
                result = test_host_tsan_sanity.main(environ)

        self.assertEqual(result, 0)
        passed_clang, passed_env = run_canaries.call_args.args
        self.assertEqual(passed_clang, clang)
        self.assertEqual(passed_env["LD_LIBRARY_PATH"], f"{runtime.parent}:/existing")
        self.assertEqual(passed_env["TSAN_OPTIONS"], test_host_tsan_sanity.TSAN_OPTIONS)
        no_gpu_nodes.assert_called_once_with()
        self.assertIn(
            "Host-TSAN clean and data-race canaries passed", output.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
