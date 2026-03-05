#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for compute_package_test_dependencies module.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

# Import after path modification
from compute_package_test_dependencies import (
    PackageInfo,
    PackageDependencyAnalyzer,
)


class PackageDependencyAnalyzerTest(unittest.TestCase):
    """Test cases for PackageDependencyAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.therock_root = Path(self.temp_dir)

        # Create math-libs directory structure
        self.math_libs_dir = self.therock_root / "math-libs" / "BLAS"
        self.math_libs_dir.mkdir(parents=True)

        # Create ml-libs directory structure
        self.ml_libs_dir = self.therock_root / "ml-libs"
        self.ml_libs_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def write_cmake_file(self, path: Path, content: str):
        """Write a CMakeLists.txt file with the given content."""
        with open(path, "w") as f:
            f.write(textwrap.dedent(content))

    def test_empty_project(self):
        """Test analyzer with no CMakeLists.txt files."""
        analyzer = PackageDependencyAnalyzer(self.therock_root)
        self.assertEqual(len(analyzer.packages), 0)
        self.assertEqual(len(analyzer.reverse_deps), 0)

    def test_single_package_no_deps(self):
        """Test parsing a single package with no dependencies."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocblas"
                BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocBLAS"
                CMAKE_ARGS
                    -DHIP_PLATFORM=amd
                COMPILER_TOOLCHAIN
                    amd-hip
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        self.assertEqual(len(analyzer.packages), 1)
        self.assertIn("rocblas", analyzer.packages)

        pkg = analyzer.packages["rocblas"]
        self.assertEqual(pkg.name, "rocblas")
        self.assertEqual(len(pkg.build_deps), 0)
        self.assertEqual(len(pkg.runtime_deps), 0)
        self.assertEqual(pkg.artifact, "math-libs")

    def test_package_with_build_deps(self):
        """Test parsing a package with BUILD_DEPS."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocblas"
                BUILD_DEPS
                    hipBLAS-common
                    rocm-cmake
                    therock-googletest
                COMPILER_TOOLCHAIN
                    amd-hip
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        pkg = analyzer.packages["rocblas"]
        self.assertEqual(len(pkg.build_deps), 3)
        self.assertIn("hipblas-common", pkg.build_deps)
        self.assertIn("rocm-cmake", pkg.build_deps)
        self.assertIn("therock-googletest", pkg.build_deps)

    def test_package_with_runtime_deps(self):
        """Test parsing a package with RUNTIME_DEPS."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rocblas"
                RUNTIME_DEPS
                    hip-clr
                    hipBLASLt
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        pkg = analyzer.packages["rocblas"]
        self.assertEqual(len(pkg.runtime_deps), 2)
        self.assertIn("hip-clr", pkg.runtime_deps)
        self.assertIn("hipblaslt", pkg.runtime_deps)

    def test_package_with_both_deps(self):
        """Test parsing a package with both BUILD_DEPS and RUNTIME_DEPS."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake
                    therock-googletest
                RUNTIME_DEPS
                    hip-clr
                    hipBLASLt
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        pkg = analyzer.packages["rocblas"]
        self.assertEqual(len(pkg.all_deps), 4)
        self.assertIn("rocm-cmake", pkg.all_deps)
        self.assertIn("hip-clr", pkg.all_deps)
        self.assertIn("hipblaslt", pkg.all_deps)

    def test_multiline_deps(self):
        """Test parsing dependencies across multiple lines."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake
                    therock-googletest
                    therock-msgpack-cxx
                RUNTIME_DEPS
                    hip-clr
                    hipBLASLt
                    ${optional_profiler_deps}
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        pkg = analyzer.packages["rocblas"]
        # Should have 5 deps (variables like ${...} are filtered out)
        self.assertEqual(len(pkg.all_deps), 5)
        self.assertIn("rocm-cmake", pkg.all_deps)
        self.assertIn("therock-msgpack-cxx", pkg.all_deps)
        self.assertIn("hipblaslt", pkg.all_deps)

    def test_deps_with_comments(self):
        """Test parsing dependencies with inline comments."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake  # CMake infrastructure
                    therock-googletest  # Testing framework
                RUNTIME_DEPS
                    hip-clr  # HIP runtime
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        pkg = analyzer.packages["rocblas"]
        self.assertEqual(len(pkg.all_deps), 3)
        self.assertIn("rocm-cmake", pkg.all_deps)
        self.assertIn("therock-googletest", pkg.all_deps)
        self.assertIn("hip-clr", pkg.all_deps)

    def test_multiple_packages_in_file(self):
        """Test parsing multiple packages from the same CMakeLists.txt."""
        cmake_content = """
            therock_cmake_subproject_declare(hipBLAS-common
                BUILD_DEPS
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    hipBLAS-common
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                BUILD_DEPS
                    hipBLAS-common
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
                    rocBLAS
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        self.assertEqual(len(analyzer.packages), 3)
        self.assertIn("hipblas-common", analyzer.packages)
        self.assertIn("rocblas", analyzer.packages)
        self.assertIn("hipblas", analyzer.packages)

    def test_reverse_dependency_graph(self):
        """Test reverse dependency graph construction."""
        cmake_content = """
            therock_cmake_subproject_declare(hipBLAS-common
                BUILD_DEPS
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    hipBLAS-common
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                BUILD_DEPS
                    hipBLAS-common
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # hipblas-common is depended on by rocblas and hipblas
        self.assertIn("rocblas", analyzer.reverse_deps["hipblas-common"])
        self.assertIn("hipblas", analyzer.reverse_deps["hipblas-common"])

        # rocblas is depended on by hipblas
        self.assertIn("hipblas", analyzer.reverse_deps["rocblas"])

        # hip-clr and rocm-cmake are not tracked (not in packages)
        self.assertNotIn("hip-clr", analyzer.reverse_deps)

    def test_transitive_dependents(self):
        """Test transitive dependent calculation."""
        cmake_content = """
            therock_cmake_subproject_declare(A
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(B
                RUNTIME_DEPS
                    A
            )

            therock_cmake_subproject_declare(C
                RUNTIME_DEPS
                    B
            )

            therock_cmake_subproject_declare(D
                RUNTIME_DEPS
                    A
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # A is depended on by B, C (transitively), and D
        transitive_deps = analyzer.get_transitive_dependents("a")
        self.assertEqual(len(transitive_deps), 3)
        self.assertIn("b", transitive_deps)
        self.assertIn("c", transitive_deps)
        self.assertIn("d", transitive_deps)

        # B is depended on by C only
        transitive_deps = analyzer.get_transitive_dependents("b")
        self.assertEqual(len(transitive_deps), 1)
        self.assertIn("c", transitive_deps)

    def test_diamond_dependency(self):
        """Test diamond dependency pattern (A <- B <- D, A <- C <- D)."""
        cmake_content = """
            therock_cmake_subproject_declare(A
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(B
                RUNTIME_DEPS
                    A
            )

            therock_cmake_subproject_declare(C
                RUNTIME_DEPS
                    A
            )

            therock_cmake_subproject_declare(D
                RUNTIME_DEPS
                    B
                    C
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # A should have all three as dependents
        transitive_deps = analyzer.get_transitive_dependents("a")
        self.assertEqual(len(transitive_deps), 3)
        self.assertIn("b", transitive_deps)
        self.assertIn("c", transitive_deps)
        self.assertIn("d", transitive_deps)

    def test_packages_to_test(self):
        """Test get_packages_to_test with multiple changed packages."""
        cmake_content = """
            therock_cmake_subproject_declare(A
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(B
                RUNTIME_DEPS
                    A
            )

            therock_cmake_subproject_declare(C
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(D
                RUNTIME_DEPS
                    C
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # If A and C change, we need to test A, B, C, D
        packages_to_test = analyzer.get_packages_to_test(["a", "c"])
        self.assertEqual(len(packages_to_test), 4)
        self.assertIn("a", packages_to_test)
        self.assertIn("b", packages_to_test)
        self.assertIn("c", packages_to_test)
        self.assertIn("d", packages_to_test)

    def test_get_dependency_info(self):
        """Test get_dependency_info returns correct structure."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    rocm-cmake
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        info = analyzer.get_dependency_info("rocblas")

        self.assertEqual(info["name"], "rocblas")
        self.assertEqual(info["artifact"], "math-libs")
        self.assertIn("rocm-cmake", info["build_dependencies"])
        self.assertIn("hip-clr", info["runtime_dependencies"])
        self.assertIn("hipblas", info["direct_dependents"])
        self.assertIn("hipblas", info["transitive_dependents"])
        self.assertEqual(info["total_dependent_count"], 1)

    def test_get_dependency_info_nonexistent(self):
        """Test get_dependency_info with nonexistent package."""
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        info = analyzer.get_dependency_info("nonexistent")

        self.assertIn("error", info)
        self.assertIn("not found", info["error"])

    def test_case_normalization(self):
        """Test that package names are normalized to lowercase."""
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                BUILD_DEPS
                    hipBLAS-common
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # All names should be lowercase
        self.assertIn("rocblas", analyzer.packages)
        self.assertIn("hipblas", analyzer.packages)
        self.assertNotIn("rocBLAS", analyzer.packages)
        self.assertNotIn("hipBLAS", analyzer.packages)

        # Dependencies should also be lowercase
        pkg = analyzer.packages["rocblas"]
        self.assertIn("hipblas-common", pkg.build_deps)
        self.assertIn("hip-clr", pkg.runtime_deps)

    def test_multiple_directories(self):
        """Test parsing packages across multiple directories."""
        # Create math-libs package
        math_cmake = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", math_cmake)

        # Create ml-libs package
        ml_cmake = """
            therock_cmake_subproject_declare(MIOpen
                RUNTIME_DEPS
                    rocBLAS
            )
        """
        self.write_cmake_file(self.ml_libs_dir / "CMakeLists.txt", ml_cmake)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        self.assertEqual(len(analyzer.packages), 2)
        self.assertIn("rocblas", analyzer.packages)
        self.assertIn("miopen", analyzer.packages)

        # Check artifacts are different
        self.assertEqual(analyzer.packages["rocblas"].artifact, "math-libs")
        self.assertEqual(analyzer.packages["miopen"].artifact, "ml-libs")

        # Check reverse deps
        self.assertIn("miopen", analyzer.reverse_deps["rocblas"])

    def test_print_test_plan_json(self):
        """Test print_test_plan with JSON output."""
        cmake_content = """
            therock_cmake_subproject_declare(A
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(B
                RUNTIME_DEPS
                    A
            )
        """
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            analyzer.print_test_plan(["a"], format="json")
            output = mock_stdout.getvalue()

        # Parse JSON output
        result = json.loads(output)
        self.assertEqual(result["changed_packages"], ["a"])
        self.assertEqual(len(result["packages_to_test"]), 2)
        self.assertIn("a", result["packages_to_test"])
        self.assertIn("b", result["packages_to_test"])
        self.assertEqual(result["test_count"], 2)

    def test_print_test_plan_text(self):
        """Test print_test_plan with text output."""
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
        cmake_file = self.math_libs_dir / "CMakeLists.txt"
        self.write_cmake_file(cmake_file, cmake_content)

        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            analyzer.print_test_plan(["rocblas"], format="text")
            output = mock_stdout.getvalue()

        # Check output contains expected strings
        self.assertIn("Changed packages: rocblas", output)
        self.assertIn("Total packages to test: 2", output)
        self.assertIn("rocblas (changed)", output)
        self.assertIn("hipblas (dependent)", output)


if __name__ == "__main__":
    unittest.main()
