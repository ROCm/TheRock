# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["THEROCK_BIN_DIR"] = str(
    Path(tempfile.gettempdir()) / "therock-rocprofiler-sdk-spm-test" / "bin"
)
sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_rocprofiler_sdk


class RocprofilerSdkSpmPreflightTest(unittest.TestCase):
    def test_preflight_script_path_matches_installed_tests_layout(self):
        expected = (
            test_rocprofiler_sdk.THEROCK_PATH
            / "share"
            / "rocprofiler-sdk"
            / "tests"
            / test_rocprofiler_sdk.ROCPROFILER_SDK_SPM_PREFLIGHT_SCRIPT
        )
        actual = (
            test_rocprofiler_sdk.ROCPROFILER_SDK_TESTS_PATH
            / test_rocprofiler_sdk.ROCPROFILER_SDK_SPM_PREFLIGHT_SCRIPT
        )
        self.assertEqual(actual, expected)

    def test_run_spm_preflight_executes_installed_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "share" / "rocprofiler-sdk" / "tests"
            tests_dir.mkdir(parents=True)
            preflight = (
                tests_dir / test_rocprofiler_sdk.ROCPROFILER_SDK_SPM_PREFLIGHT_SCRIPT
            )
            preflight.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

            with mock.patch.object(
                test_rocprofiler_sdk, "ROCPROFILER_SDK_TESTS_PATH", tests_dir
            ), mock.patch.object(
                test_rocprofiler_sdk.subprocess, "run", autospec=True
            ) as run:
                test_rocprofiler_sdk.run_spm_preflight()

            run.assert_called_once()
            command, kwargs = run.call_args
            self.assertEqual(command[0], [sys.executable, str(preflight)])
            self.assertEqual(kwargs["cwd"], tests_dir)
            self.assertTrue(kwargs["check"])

    def test_run_spm_preflight_fails_when_script_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            with mock.patch.object(
                test_rocprofiler_sdk, "ROCPROFILER_SDK_TESTS_PATH", tests_dir
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    test_rocprofiler_sdk.run_spm_preflight()
                self.assertIn(
                    test_rocprofiler_sdk.ROCPROFILER_SDK_SPM_PREFLIGHT_SCRIPT,
                    str(ctx.exception),
                )


if __name__ == "__main__":
    unittest.main()
