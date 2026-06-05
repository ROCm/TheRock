# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import unittest
from pathlib import Path

THEROCK_DIR = Path(__file__).parent.parent.parent
SCRIPT = Path(__file__).parent.parent / "determine_rocm_test_dependencies.py"

sys.path.insert(0, str(THEROCK_DIR / "test_tools"))

from determine_rocm_test_dependencies import (
    parse_cmake_test_subprojects,
    get_subprojects_to_test,
)


class TestDetermineRocmTestDependencies(unittest.TestCase):
    def test_parse_cmake_test_subprojects(self):
        """Parse CMakeLists.txt and verify key dependencies."""
        test_deps = parse_cmake_test_subprojects(THEROCK_DIR)

        self.assertEqual(set(test_deps["rocblas"]), {"hipblas", "rocsolver"})
        self.assertEqual(test_deps["rocwmma"], [])  # empty TEST_SUBPROJECTS

    def test_get_subprojects_to_test(self):
        """Test dependency resolution with case-insensitive input."""
        result = get_subprojects_to_test(["rocBLAS"], THEROCK_DIR)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

        # Path format normalization (projects/rocblas -> rocblas)
        result = get_subprojects_to_test([Path("projects/rocblas").name], THEROCK_DIR)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

    def test_empty_changed_projects_outputs_wildcard(self):
        """Empty --changed-projects outputs '*' for all tests."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.stdout.strip(), "*")

    def test_empty_changed_projects_flag_outputs_wildcard(self):
        """Empty --changed-projects flag outputs '*' for all tests."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--changed-projects"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "*")


if __name__ == "__main__":
    unittest.main()
