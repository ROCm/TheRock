# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
import unittest
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parents[1]


class TsanHipblasltIntegrationTest(unittest.TestCase):
    def test_tsan_enables_upstream_codegen_launcher(self):
        source = (
            THEROCK_DIR / "math-libs" / "BLAS" / "pre_hook_hipBLASLt.cmake"
        ).read_text(encoding="utf-8")

        tsan_block = re.search(
            r'if\(THEROCK_SANITIZER STREQUAL "TSAN"\)(.*?)endif\(\)',
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(tsan_block)
        self.assertRegex(
            tsan_block.group(1),
            r"set\(HIPBLASLT_ENABLE_TSAN ON CACHE BOOL\s+" r'"[^"]+" FORCE\)',
        )

    def test_does_not_replace_global_sanitizer_configuration(self):
        source = (
            THEROCK_DIR / "math-libs" / "BLAS" / "pre_hook_hipBLASLt.cmake"
        ).read_text(encoding="utf-8")

        # The pre-hook opts into hipBLASLt's subprocess launcher only. Compile
        # and link instrumentation remains owned by TheRock's global sanitizer
        # setup, which avoids adding unscoped device-side TSAN flags.
        self.assertNotIn("add_compile_options", source)
        self.assertNotIn("add_link_options", source)
        self.assertNotIn("CMAKE_CXX_FLAGS", source)


if __name__ == "__main__":
    unittest.main()
