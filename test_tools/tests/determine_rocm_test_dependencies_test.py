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
    list_subprojects,
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
        self.assertEqual(
            set(test_deps["rocsparse"]), {"hipsparse", "rocsolver", "hipsolver"}
        )

        # Verify hipSPARSE test dependencies
        self.assertIn("hipsparse", test_deps)
        self.assertEqual(set(test_deps["hipsparse"]), {"hipsparse"})

        # Verify hipSPARSELt test dependencies
        self.assertIn("hipsparselt", test_deps)
        self.assertEqual(set(test_deps["hipsparselt"]), {"hipsparselt"})

        # Verify rocRAND test dependencies
        self.assertIn("rocrand", test_deps)
        self.assertEqual(set(test_deps["rocrand"]), {"hiprand"})

        # Verify rocPRIM test dependencies
        self.assertIn("rocprim", test_deps)
        self.assertEqual(
            set(test_deps["rocprim"]), {"hipcub", "rocthrust", "rocsparse"}
        )

        # Verify rocFFT test dependencies
        self.assertIn("rocfft", test_deps)
        self.assertEqual(set(test_deps["rocfft"]), {"hipfft"})

        # Verify rocWMMA has empty TEST_SUBPROJECTS (tests only itself)
        self.assertIn("rocwmma", test_deps)
        self.assertEqual(test_deps["rocwmma"], [])

        # Verify hipRAND has empty TEST_SUBPROJECTS (tests only itself)
        self.assertIn("hiprand", test_deps)
        self.assertEqual(test_deps["hiprand"], [])

        # Verify hipFFT has empty TEST_SUBPROJECTS (tests only itself)
        self.assertIn("hipfft", test_deps)
        self.assertEqual(test_deps["hipfft"], [])

    def test_get_subprojects_to_test(self):
        """Test get_subprojects_to_test with real CMake files."""
        therock_dir = Path(__file__).parent.parent.parent

        # Test rocBLAS (input is case-insensitive, output is lowercase)
        result = get_subprojects_to_test(["rocBLAS"], therock_dir)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

        # Test rocSPARSE
        result = get_subprojects_to_test(["ROCSPARSE"], therock_dir)
        self.assertEqual(result, {"rocsparse", "hipsparse", "rocsolver", "hipsolver"})

    def test_empty_test_subprojects(self):
        """Test that empty TEST_SUBPROJECTS works correctly."""
        therock_dir = Path(__file__).parent.parent.parent

        # rocWMMA has empty TEST_SUBPROJECTS, should only return itself
        result = get_subprojects_to_test(["rocWMMA"], therock_dir)
        self.assertEqual(result, {"rocwmma"})

        # hipRAND has empty TEST_SUBPROJECTS, should only return itself
        result = get_subprojects_to_test(["hipRAND"], therock_dir)
        self.assertEqual(result, {"hiprand"})

        # hipFFT has empty TEST_SUBPROJECTS, should only return itself
        result = get_subprojects_to_test(["hipFFT"], therock_dir)
        self.assertEqual(result, {"hipfft"})

    def test_list_subprojects_with_show_deps(self):
        """Test list_subprojects with show_deps option."""
        therock_dir = Path(__file__).parent.parent.parent

        # Without show_deps, should return list
        result = list_subprojects(therock_dir, show_deps=False)
        self.assertIsInstance(result, list)
        self.assertIn("rocblas", result)
        self.assertIn("rocwmma", result)

        # With show_deps, should return dict with "empty" for empty deps
        result = list_subprojects(therock_dir, show_deps=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["rocblas"], ["hipblas", "rocsolver"])
        self.assertEqual(result["rocwmma"], "empty")
        self.assertEqual(result["hiprand"], "empty")
        self.assertEqual(result["hipfft"], "empty")


if __name__ == "__main__":
    unittest.main()
