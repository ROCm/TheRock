# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_target_run


class ConfigureTargetRunTest(unittest.TestCase):
    def test_linux_gfx94X(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94x", "linux")
        self.assertEqual(runner_label, "linux-gfx942-1gpu-ossci-rocm")

    def test_linux_gfx94X_dcgpu(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94X-dcgpu", "linux")
        self.assertEqual(runner_label, "linux-gfx942-1gpu-ossci-rocm")

    def test_windows_gfx115x(self):
        runner_label = configure_target_run.get_runner_label("gfx1151", "windows")
        self.assertEqual(runner_label, "windows-gfx1151-gpu-rocm")

    def test_windows_gfx1151_pytorch_ci_runner(self):
        ref = "refs/heads/main/.github/workflows/build_windows_pytorch_wheels.yml"
        with patch.dict(os.environ, {"GITHUB_WORKFLOW_REF": ref}, clear=False):
            runner_label = configure_target_run.get_runner_label("gfx1151", "windows")
        self.assertEqual(runner_label, "windows-strix-halo-gpu-rocm-128gb")

    def test_windows_gfx120X_all(self):
        runner_label = configure_target_run.get_runner_label("gfx120X-all", "windows")
        # No runner label yet.
        self.assertEqual(runner_label, "")


if __name__ == "__main__":
    unittest.main()
