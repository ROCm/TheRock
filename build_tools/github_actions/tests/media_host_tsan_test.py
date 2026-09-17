# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the device-free media host-TSAN runner."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_executable_scripts"))

import test_media_host_tsan


class MediaHostTsanTest(unittest.TestCase):
    def test_exact_inventories(self):
        for component, config in test_media_host_tsan.COMPONENTS.items():
            names = list(reversed(config["expected_names"]))
            self.assertEqual(
                test_media_host_tsan._validate_inventory(component, names),
                tuple(sorted(config["expected_names"])),
            )

    def test_inventory_change_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_media_host_tsan._validate_inventory(
                "rocjpeg", ["rocjpeg.stream.mug_400"]
            )

    def test_compile_command_directly_enables_tsan(self):
        command = test_media_host_tsan._compile_command(
            Path("/rocm/llvm/bin/clang++"),
            Path("/src/media.cpp"),
            Path("/tmp/media-test"),
            Path("/rocm"),
            "rocdecode",
        )
        self.assertIn("-fsanitize=thread", command)
        self.assertIn("-shared-libsan", command)
        self.assertIn("-D__HIP_PLATFORM_AMD__=1", command)
        self.assertIn("-fno-omit-frame-pointer", command)
        self.assertIn("-lrocdecode", command)
        self.assertNotIn("LD_PRELOAD", " ".join(command))


if __name__ == "__main__":
    unittest.main()
