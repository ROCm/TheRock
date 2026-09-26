# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parents[1]


class TsanModuleRpathTest(unittest.TestCase):
    def test_sanitizer_runtime_rpath_covers_loadable_modules(self):
        source = (
            THEROCK_DIR / "cmake" / "therock_global_post_subproject.cmake"
        ).read_text(encoding="utf-8")

        # The target inventory must classify CMake MODULE libraries separately;
        # otherwise native Python extensions such as hipBLASLt's _rocisa are
        # silently omitted from all subsequent processing.
        self.assertRegex(
            source,
            r'elseif\("\$\{target_type\}" STREQUAL "MODULE_LIBRARY"\)\s*'
            r'list\(APPEND THEROCK_MODULE_TARGETS "\$\{target\}"\)',
        )

        self.assertIn(
            'target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY"',
            source,
        )

        # Restrict this assertion to the RPATH loop. MODULE targets also appear
        # in the split-debug-info loop, which must not let this regression test
        # pass when RPATH processing accidentally drops them.
        rpath_section = source.split(
            "# Iterate over all dynamically linked targets", maxsplit=1
        )[1].split("# Process all dynamically linked targets", maxsplit=1)[0]
        self.assertRegex(
            rpath_section,
            r"foreach\(target[\s\S]*\$\{THEROCK_EXECUTABLE_TARGETS\}"
            r"[\s\S]*\$\{THEROCK_SHARED_LIBRARY_TARGETS\}"
            r"[\s\S]*\$\{THEROCK_MODULE_TARGETS\}\)",
        )


if __name__ == "__main__":
    unittest.main()
