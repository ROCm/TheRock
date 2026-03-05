#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Basic sanity tests for compute_package_test_dependencies module.
"""

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from compute_package_test_dependencies import (
    PackageInfo,
    PackageDependencyAnalyzer,
)


class PackageDependencyAnalyzerTest(unittest.TestCase):
    """Basic sanity tests for PackageDependencyAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.therock_root = Path(self.temp_dir)
        self.math_libs_dir = self.therock_root / "math-libs" / "BLAS"
        self.math_libs_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def write_cmake_file(self, path: Path, content: str):
        """Write a CMakeLists.txt file."""
        with open(path, "w") as f:
            f.write(textwrap.dedent(content))

    def test_basic_parsing(self):
        """Sanity check: can parse a basic CMakeLists.txt."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Just check something was parsed
        self.assertGreater(len(analyzer.packages), 0)
        self.assertIn("rocblas", analyzer.packages)

    def test_dependency_tracking(self):
        """Sanity check: dependencies are tracked."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Check that some dependencies were found
        rocblas_pkg = analyzer.packages.get("rocblas")
        self.assertIsNotNone(rocblas_pkg)
        self.assertGreater(len(rocblas_pkg.all_deps), 0)

    def test_reverse_dependencies(self):
        """Sanity check: reverse dependencies are computed."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Check that rocblas has reverse dependencies
        self.assertIn("rocblas", analyzer.reverse_deps)
        self.assertGreater(len(analyzer.reverse_deps["rocblas"]), 0)

    def test_packages_to_test(self):
        """Sanity check: get_packages_to_test returns something."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        packages = analyzer.get_packages_to_test(["rocblas"])
        self.assertGreater(len(packages), 0)
        self.assertIn("rocblas", packages)

    def test_dependency_info(self):
        """Sanity check: get_dependency_info returns data."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        info = analyzer.get_dependency_info("rocblas")
        self.assertIn("name", info)
        self.assertEqual(info["name"], "rocblas")


if __name__ == "__main__":
    unittest.main()
