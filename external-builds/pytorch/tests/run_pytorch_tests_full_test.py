# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the run_test.py command built by run_pytorch_tests_full.py."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Prefer the PyTorch tree so importing the runner does not bind JAX's
# identically named skip_tests package from other collected tests.
PYTORCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.fspath(PYTORCH_DIR))
for _name in list(sys.modules):
    if _name == "skip_tests" or _name.startswith("skip_tests."):
        del sys.modules[_name]

from run_pytorch_tests_full import build_run_test_cmd

OLD_RUN_TEST = """
parser.add_argument(
    "--exclude-jit-executor",
    action="store_true",
)
"""
NEW_RUN_TEST = """
parser.add_argument(
    "--exclude-torch-export-tests",
    action="store_true",
)
"""


class BuildRunTestCmdTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pytorch_dir = Path(self.temp_dir.name)
        (self.pytorch_dir / "test").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_cmd(self, run_test_source: str, exclude_jit_executor: bool = True):
        (self.pytorch_dir / "test" / "run_test.py").write_text(run_test_source)
        args = argparse.Namespace(
            pytorch_dir=self.pytorch_dir,
            exclude_jit_executor=exclude_jit_executor,
            exclude_distributed=True,
            exclude_quantization=True,
            test_config="default",
            dry_run=False,
            shard=0,
            num_shards=0,
            include=None,
            exclude=None,
            cache=True,
        )
        return build_run_test_cmd(args, tests_to_skip="", passthrough_args=[])

    def test_passes_flag_when_run_test_supports_it(self):
        cmd = self._build_cmd(OLD_RUN_TEST)
        self.assertIn("--exclude-jit-executor", cmd)

    def test_omits_flag_when_run_test_removed_it(self):
        cmd = self._build_cmd(NEW_RUN_TEST)
        self.assertNotIn("--exclude-jit-executor", cmd)
        self.assertIn("--exclude-distributed-tests", cmd)
        self.assertIn("--exclude-quantization-tests", cmd)

    def test_omits_flag_when_disabled(self):
        cmd = self._build_cmd(OLD_RUN_TEST, exclude_jit_executor=False)
        self.assertNotIn("--exclude-jit-executor", cmd)


if __name__ == "__main__":
    unittest.main()
