# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the common PyTorch wheel build environment."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import _setup_common_build_env


class CommonBuildEnvironmentTest(unittest.TestCase):
    def test_rocm_paths_are_discovered_by_pytorch_cmake(self):
        env = _setup_common_build_env(
            Path("/tmp/rocm/bin"),
            Path("/tmp/rocm"),
            "gfx942",
            None,
            False,
        )

        self.assertNotIn("CMAKE_PREFIX_PATH", env)
        self.assertNotIn("ROCM_HOME", env)
        self.assertNotIn("ROCM_PATH", env)
        self.assertEqual(env["PYTORCH_ROCM_ARCH"], "gfx942")
        self.assertTrue(env["PATH"].startswith("/tmp/rocm/bin" + os.path.pathsep))


if __name__ == "__main__":
    unittest.main()
