# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTestUtils(unittest.TestCase):
    def test_get_ctest_junit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ, {"OUTPUT_ARTIFACTS_DIR": tmpdir, "SHARD_INDEX": "2"}
            ):
                import importlib
                import test_utils

                importlib.reload(test_utils)
                path = test_utils.get_ctest_junit_path("hipblas")

                self.assertEqual(path.name, "ctest-hipblas-shard2.xml")

    def test_get_gtest_output_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ, {"OUTPUT_ARTIFACTS_DIR": tmpdir, "SHARD_INDEX": "1"}
            ):
                import importlib
                import test_utils

                importlib.reload(test_utils)
                arg = test_utils.get_gtest_output_arg("rocblas")

                self.assertTrue(arg.startswith("--gtest_output=json:"))
                self.assertIn("gtest-rocblas-shard1.json", arg)


if __name__ == "__main__":
    unittest.main()
