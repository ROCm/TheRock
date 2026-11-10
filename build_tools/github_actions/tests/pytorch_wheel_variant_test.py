# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import pytorch_wheel_variant
from pytorch_wheel_variant import plan_for


class TorchWheelPlanTest(unittest.TestCase):
    def test_release_keeps_companions(self):
        plan = plan_for("release")

        self.assertFalse(plan.build_torch_only)
        self.assertEqual(plan.build_variant_arg, "")
        self.assertTrue(plan.split_torchvision)
        self.assertEqual(plan.torch_only_arg, "")

    def test_empty_variant_is_a_normal_build(self):
        plan = plan_for("  ")

        self.assertFalse(plan.build_torch_only)
        self.assertEqual(plan.build_variant_arg, "")

    def test_asan_variants_build_torch_only(self):
        for variant in ("asan", "asan-debug", "host-asan", "host-asan-debug"):
            with self.subTest(variant=variant):
                plan = plan_for(variant)

                self.assertTrue(plan.build_torch_only)
                self.assertEqual(plan.build_variant_arg, f"--build-variant={variant}")
                self.assertFalse(plan.split_torchvision)
                self.assertEqual(plan.torch_only_arg, "--torch-only")

    def test_tsan_is_not_an_asan_build(self):
        plan = plan_for("tsan")

        self.assertFalse(plan.build_torch_only)
        self.assertEqual(plan.build_variant_arg, "")

    def test_variant_with_whitespace_is_rejected(self):
        for variant in ("asan debug", "asan\nFOO=bar"):
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    plan_for(variant)

    def test_github_env_records_a_host_asan_plan(self):
        plan = plan_for("host-asan")
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "github_env"

            pytorch_wheel_variant.write_github_env(plan, os.fspath(env_file))
            text = env_file.read_text(encoding="utf-8")

        self.assertIn("PYTORCH_BUILD_TORCH_ONLY=true\n", text)
        self.assertIn("PYTORCH_BUILD_VARIANT_ARG=--build-variant=host-asan\n", text)
        self.assertIn("PYTORCH_SPLIT_TORCHVISION=false\n", text)
        self.assertIn("PYTORCH_TORCH_ONLY_ARG=--torch-only\n", text)


if __name__ == "__main__":
    unittest.main()
