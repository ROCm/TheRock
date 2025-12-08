import json
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_target_run

therock_test_runner_dict = {
    "gfx94x": {"linux": "linux-mi325-1gpu-ossci-rocm-frac"},
    "gfx110x": {
        "linux": "linux-gfx110X-gpu-rocm",
        "windows": "windows-gfx110X-gpu-rocm",
    },
    "gfx1151": {
        "linux": "linux-strix-halo-gpu-rocm",
        "windows": "windows-strix-halo-gpu-rocm",
    },
    "gfx950": {"linux": ""},
    "gfx120x": {"linux": "linux-rx9070-gpu-rocm", "windows": ""},
    "gfx90x": {"linux": "", "windows": ""},
    "gfx101x": {"linux": "", "windows": ""},
    "gfx103x": {
        "linux": "linux-rx6950-gpu-rocm",
        "windows": "windows-gfx1030-gpu-rocm",
    },
    "gfx1150": {"linux": "", "windows": ""},
    "gfx1152": {"linux": "", "windows": ""},
    "gfx1153": {"linux": "", "windows": ""},
}

os.environ["ROCM_THEROCK_TEST_RUNNERS"] = json.dumps(therock_test_runner_dict)


class ConfigureTargetRunTest(unittest.TestCase):
    def test_linux_gfx94X(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94x", "linux")
        self.assertEqual(runner_label, "linux-mi325-1gpu-ossci-rocm-frac")

    def test_linux_gfx94X_dcgpu(self):
        # gfx94x is the outer key used to construct workflow pipelines, while
        # gfx94X-dcgpu is the inner key, which we use for package names. When
        # run from a workflow, we expect to only work on the inner keys.
        runner_label = configure_target_run.get_runner_label("gfx94X-dcgpu", "linux")
        self.assertEqual(runner_label, "linux-mi325-1gpu-ossci-rocm-frac")

    def test_windows_gfx115x(self):
        runner_label = configure_target_run.get_runner_label("gfx1151", "windows")
        self.assertEqual(runner_label, "windows-strix-halo-gpu-rocm")

    def test_windows_gfx120X_all(self):
        runner_label = configure_target_run.get_runner_label("gfx120X-all", "windows")
        # No runner label yet.
        self.assertEqual(runner_label, "")


if __name__ == "__main__":
    unittest.main()
