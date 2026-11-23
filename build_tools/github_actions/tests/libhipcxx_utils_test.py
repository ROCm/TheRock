# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import libhipcxx_utils


def _offload_arch_prints(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


class TryOffloadArchTest(unittest.TestCase):
    def test_plain_gfx_name_is_returned(self):
        with patch("subprocess.run", return_value=_offload_arch_prints("gfx950\n")):
            self.assertEqual(libhipcxx_utils._try_offload_arch("/build", ""), "gfx950")

    def test_target_features_are_dropped(self):
        with patch(
            "subprocess.run",
            return_value=_offload_arch_prints("gfx950:sramecc+:xnack-\n"),
        ):
            self.assertEqual(libhipcxx_utils._try_offload_arch("/build", ""), "gfx950")


if __name__ == "__main__":
    unittest.main()
