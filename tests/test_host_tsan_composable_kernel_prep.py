# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
import unittest
import json
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parents[1]


class HostTsanComposableKernelPreparationTest(unittest.TestCase):
    def test_debug_info_is_removed_only_for_host_tsan(self):
        source = (
            THEROCK_DIR / "cmake" / "therock_host_tsan_device_build.cmake"
        ).read_text(encoding="utf-8")
        self.assertIn('THEROCK_SANITIZER STREQUAL "HOST_TSAN"', source)
        self.assertIn("CMAKE_C_FLAGS_INIT CMAKE_CXX_FLAGS_INIT", source)
        self.assertIn("CMAKE_C_FLAGS CMAKE_CXX_FLAGS", source)
        self.assertIn("DEFINED CACHE{${_flags_var}}", source)
        self.assertIn(')-g([0-9]+)?', source)
        self.assertIn(')-gdwarf-[0-9]+', source)

    def test_only_known_device_heavy_dependencies_use_the_helper(self):
        for hook_name in (
            "pre_hook_composable_kernel.cmake",
            "pre_hook_hipTensor.cmake",
        ):
            source = (THEROCK_DIR / "math-libs" / hook_name).read_text(
                encoding="utf-8"
            )
            self.assertIn("therock_host_tsan_device_build.cmake", source)
            self.assertIn("therock_host_tsan_strip_debug_info()", source)

    def test_device_heavy_dependencies_use_release_builds(self):
        presets = json.loads((THEROCK_DIR / "CMakePresets.json").read_text())
        host_tsan = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "linux-host-tsan-base"
        )
        cache = host_tsan["cacheVariables"]
        self.assertEqual(cache["composable_kernel_BUILD_TYPE"], "Release")
        self.assertEqual(cache["hipTensor_BUILD_TYPE"], "Release")

    def test_host_tsan_instrumentation_is_not_removed(self):
        source = (
            THEROCK_DIR / "cmake" / "therock_host_tsan_device_build.cmake"
        ).read_text(encoding="utf-8")
        self.assertNotIn("fsanitize", source)
        self.assertNotIn("Xarch_host", source)
        self.assertNotIn("THEROCK_SANITIZER OFF", source)


if __name__ == "__main__":
    unittest.main()
