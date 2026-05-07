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

        # Verify rocBLAS test dependencies (all lowercase)
        self.assertIn("rocblas", test_deps)
        self.assertEqual(set(test_deps["rocblas"]), {"hipblas", "rocsolver"})

        # Verify rocSPARSE test dependencies
        self.assertIn("rocsparse", test_deps)
        self.assertEqual(set(test_deps["rocsparse"]), {"rocsparse", "hipsparse"})

        # Verify hipSPARSE test dependencies
        self.assertIn("hipsparse", test_deps)
        self.assertEqual(set(test_deps["hipsparse"]), {"hipsparse"})

        # Verify hipSPARSELt test dependencies
        self.assertIn("hipsparselt", test_deps)
        self.assertEqual(set(test_deps["hipsparselt"]), {"hipsparselt"})

    def test_get_subprojects_to_test(self):
        """Test get_subprojects_to_test with real CMake files."""
        therock_dir = Path(__file__).parent.parent.parent

        # Test rocBLAS (input is case-insensitive, output is lowercase)
        result = get_subprojects_to_test(["rocBLAS"], therock_dir)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

        # Test rocSPARSE
        result = get_subprojects_to_test(["ROCSPARSE"], therock_dir)
        self.assertEqual(result, {"rocsparse", "hipsparse"})


if __name__ == "__main__":
    unittest.main()
