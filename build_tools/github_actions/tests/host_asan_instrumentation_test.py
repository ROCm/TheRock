# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))
import host_asan_instrumentation
import test_host_asan_sanity


class HostAsanInstrumentationTest(unittest.TestCase):
    def test_native_environment_removes_preload_without_mutating_source(self):
        source = {
            "LD_PRELOAD": "/tmp/libclang_rt.asan.so",
            "ASAN_OPTIONS": "detect_leaks=1",
        }
        env = host_asan_instrumentation.native_host_asan_environment(source)

        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["ASAN_OPTIONS"], "detect_leaks=1")
        self.assertIn("LD_PRELOAD", source)

    def test_accepts_direct_clang_asan_needed_entry_without_preload(self):
        executable = Path("instrumented")
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                " 0x0000000000000001 (NEEDED)             Shared library: "
                "[libclang_rt.asan-x86_64.so]\n"
            ),
            stderr="",
        )
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                host_asan_instrumentation.subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            host_asan_instrumentation.require_direct_clang_asan(
                executable, {"LD_PRELOAD": "/tmp/not-allowed.so"}
            )

        self.assertNotIn("LD_PRELOAD", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.args[0], ["readelf", "-d", str(executable)])

    def test_rejects_executable_without_direct_clang_asan(self):
        executable = Path("uninstrumented")
        completed = subprocess.CompletedProcess(
            [], 0, stdout="NEEDED libc.so.6\n", stderr=""
        )
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                host_asan_instrumentation.subprocess,
                "run",
                return_value=completed,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "not directly linked"):
                host_asan_instrumentation.require_direct_clang_asan(executable, {})

    def test_rejects_missing_executable_before_readelf(self):
        with patch.object(host_asan_instrumentation.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "executable is missing"):
                host_asan_instrumentation.require_direct_clang_asan(
                    Path("missing-host-asan-binary"), {}
                )
        run.assert_not_called()

    def test_sanity_checks_canary_and_executes_without_preload(self):
        compile_result = subprocess.CompletedProcess([], 0)
        canary_result = subprocess.CompletedProcess(
            [],
            -6,
            stdout="",
            stderr="ERROR: AddressSanitizer: heap-use-after-free",
        )
        with (
            patch.dict(
                os.environ,
                {
                    "THEROCK_BIN_DIR": "/tmp/rocm/bin",
                    "ASAN_RUNTIME_PATH": "/tmp/libclang_rt.asan.so",
                    "LD_PRELOAD": "/tmp/not-allowed.so",
                },
                clear=False,
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "write_text"),
            patch.object(test_host_asan_sanity.tempfile, "TemporaryDirectory") as tmp,
            patch.object(test_host_asan_sanity, "require_direct_clang_asan") as require,
            patch.object(
                test_host_asan_sanity.subprocess,
                "run",
                side_effect=(compile_result, canary_result),
            ) as run,
        ):
            tmp.return_value.__enter__.return_value = "/tmp/canary"
            self.assertEqual(test_host_asan_sanity.main(), 0)

        self.assertEqual(require.call_args.args[0].name, "canary")
        self.assertNotIn("LD_PRELOAD", require.call_args.args[1])
        self.assertNotIn("LD_PRELOAD", run.call_args.kwargs["env"])

if __name__ == "__main__":
    unittest.main()
