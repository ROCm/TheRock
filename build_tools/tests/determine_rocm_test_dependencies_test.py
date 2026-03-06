# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import (
    PackageDependencyAnalyzer,
    create_analyzer,
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
                RUNTIME_DEPS
                    hip-clr
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        self.assertIn("rocblas", analyzer.packages)

    def test_get_packages_to_test(self):
        """Sanity check: get_packages_to_test returns expected packages."""
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
        self.assertIn("rocblas", packages)
        self.assertIn("hipblas", packages)

    def test_create_analyzer(self):
        """Sanity check: create_analyzer helper works."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        analyzer = create_analyzer(self.therock_root)

        self.assertIn("rocblas", analyzer.packages)


if __name__ == "__main__":
    unittest.main()
