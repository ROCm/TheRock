# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import (
    parse_cmake_test_subprojects,
    get_subprojects_to_test,
)


class CMakeParserTest(unittest.TestCase):
    def test_parse_real_cmake_files(self):
        """Parse actual CMakeLists.txt files in the repo."""
        therock_dir = Path(__file__).parent.parent.parent
        test_deps = parse_cmake_test_subprojects(therock_dir)

        # Verify rocBLAS test dependencies
        self.assertIn("rocBLAS", test_deps)
        self.assertEqual(set(test_deps["rocBLAS"]), {"hipBLAS", "rocSOLVER"})

        # Verify rocSPARSE test dependencies
        self.assertIn("rocSPARSE", test_deps)
        self.assertEqual(set(test_deps["rocSPARSE"]), {"rocSPARSE", "hipSPARSE"})

        # Verify hipSPARSE test dependencies
        self.assertIn("hipSPARSE", test_deps)
        self.assertEqual(set(test_deps["hipSPARSE"]), {"hipSPARSE"})

        # Verify hipSPARSELt test dependencies
        self.assertIn("hipSPARSELt", test_deps)
        self.assertEqual(set(test_deps["hipSPARSELt"]), {"hipSPARSELt"})

    def test_get_subprojects_to_test(self):
        """Test get_subprojects_to_test with real CMake files."""
        therock_dir = Path(__file__).parent.parent.parent

        # Test rocBLAS
        result = get_subprojects_to_test(["rocBLAS"], therock_dir)
        self.assertEqual(result, {"rocBLAS", "hipBLAS", "rocSOLVER"})

        # Test rocSPARSE
        result = get_subprojects_to_test(["rocSPARSE"], therock_dir)
        self.assertEqual(result, {"rocSPARSE", "hipSPARSE"})


if __name__ == "__main__":
    unittest.main()
