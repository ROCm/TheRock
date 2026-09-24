# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parent.parent


class TsanWorkflowEntrypointsTest(unittest.TestCase):
    def test_entry_points_use_the_regular_comprehensive_gpu_suite(self):
        workflows = THEROCK_DIR / ".github" / "workflows"
        for name in ("multi_arch_ci_tsan.yml", "multi_arch_release_tsan.yml"):
            with self.subTest(workflow=name):
                workflow = (workflows / name).read_text(encoding="utf-8")
                self.assertIn("test_type: comprehensive", workflow)

        linux = (workflows / "multi_arch_ci_linux.yml").read_text(encoding="utf-8")
        self.assertIn("test_artifacts_per_family:", linux)
        self.assertNotIn("test_host_sanitizer_cpu:", linux)


if __name__ == "__main__":
    unittest.main()
