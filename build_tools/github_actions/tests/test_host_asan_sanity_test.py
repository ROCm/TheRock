# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))
import test_host_asan_sanity


class HostAsanSanityTest(unittest.TestCase):
    @patch.object(Path, "write_text")
    @patch.object(Path, "is_file", return_value=True)
    @patch.object(test_host_asan_sanity.tempfile, "TemporaryDirectory")
    @patch.object(test_host_asan_sanity.subprocess, "run")
    def test_expected_asan_failure_passes(
        self, run, temporary_directory, _is_file, _write_text
    ):
        temporary_directory.return_value.__enter__.return_value = "/tmp/canary"
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess(
                [],
                -6,
                stdout="",
                stderr="ERROR: AddressSanitizer: heap-use-after-free",
            ),
        ]
        with patch.dict(
            os.environ,
            {
                "THEROCK_BIN_DIR": "/tmp/rocm/bin",
                "ASAN_RUNTIME_PATH": "/tmp/libclang_rt.asan.so",
            },
            clear=False,
        ):
            self.assertEqual(test_host_asan_sanity.main(), 0)

    def test_missing_compiler_fails(self):
        with patch.dict(
            os.environ,
            {
                "THEROCK_BIN_DIR": "/tmp/rocm/bin",
                "ASAN_RUNTIME_PATH": "/tmp/libclang_rt.asan.so",
            },
            clear=False,
        ):
            self.assertEqual(test_host_asan_sanity.main(), 1)


if __name__ == "__main__":
    unittest.main()
