from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_target_run


class ConfigureTargetRunTest(unittest.TestCase):
    def test_linux_gfx94X(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94x", "linux")
        self.assertEqual(runner_label, "linux-mi325-1gpu-ossci-rocm")

    def test_linux_gfx94X_dcgpu(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94X-dcgpu", "linux")
        self.assertEqual(runner_label, "linux-mi325-1gpu-ossci-rocm")

    def test_windows_gfx115x(self):
        runner_label = configure_target_run.get_runner_label("gfx115x", "windows")
        self.assertEqual(runner_label, "windows-strix-halo-gpu-rocm")

    def test_windows_gfx1151(self):
        runner_label = configure_target_run.get_runner_label("gfx1151", "windows")
        self.assertEqual(runner_label, "windows-strix-halo-gpu-rocm")

    def test_windows_gfx120X_all(self):
        runner_label = configure_target_run.get_runner_label("gfx120X-all", "windows")
        # No runner label yet.
        self.assertEqual(runner_label, "")


if __name__ == "__main__":
    unittest.main()
