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
        # Use default (include_runtime_deps=True) which should pick up hipBLAS
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # Check that rocblas has reverse dependencies (hipblas has RUNTIME_DEPS on it)
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

    def test_runtime_deps_vs_build_deps(self):
        """
        Test that RUNTIME_DEPS vs BUILD_DEPS filtering works correctly.

        By default: only RUNTIME_DEPS trigger testing (packages that actually use the dependency)
        With include_build_deps: BUILD_DEPS also trigger testing
        """
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )

            therock_cmake_subproject_declare(rocSPARSE
                BUILD_DEPS
                    rocBLAS
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)

        # Default: only RUNTIME_DEPS (include_runtime_deps=True, include_build_deps=False)
        analyzer_runtime_only = PackageDependencyAnalyzer(
            self.therock_root, include_build_deps=False, include_runtime_deps=True
        )
        packages = analyzer_runtime_only.get_packages_to_test(["rocblas"])
        self.assertIn("rocblas", packages)
        self.assertIn("hipblas", packages, "hipblas has RUNTIME_DEPS on rocblas")
        self.assertNotIn("rocsparse", packages, "rocsparse only has BUILD_DEPS, should be excluded")

        # With BUILD_DEPS included
        analyzer_both = PackageDependencyAnalyzer(
            self.therock_root, include_build_deps=True, include_runtime_deps=True
        )
        packages = analyzer_both.get_packages_to_test(["rocblas"])
        self.assertIn("rocblas", packages)
        self.assertIn("hipblas", packages, "hipblas has RUNTIME_DEPS on rocblas")
        self.assertIn("rocsparse", packages, "rocsparse has BUILD_DEPS on rocblas and should be included")

        # Only BUILD_DEPS
        analyzer_build_only = PackageDependencyAnalyzer(
            self.therock_root, include_build_deps=True, include_runtime_deps=False
        )
        packages = analyzer_build_only.get_packages_to_test(["rocblas"])
        self.assertIn("rocblas", packages)
        self.assertNotIn("hipblas", packages, "hipblas has RUNTIME_DEPS, should be excluded")
        self.assertIn("rocsparse", packages, "rocsparse has BUILD_DEPS on rocblas")

    def test_only_downstream_tested(self):
        """
        CRITICAL TEST: Verify we only test DIRECT downstream, NOT upstream or transitive dependencies.

        Dependency chain: rocblas ← hipblas ← hipblaslt
        (hipblas depends on rocblas, hipblaslt depends on hipblas)
        """
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )

            therock_cmake_subproject_declare(hipBLASLt
                RUNTIME_DEPS
                    hipBLAS
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        # Use default (include_runtime_deps=True, include_build_deps=False)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # When rocblas changes: test rocblas and hipblas (DIRECT dependent only)
        # Do NOT test hipblaslt (transitive - depends on hipblas, not rocblas directly)
        packages = analyzer.get_packages_to_test(["rocblas"])
        self.assertIn("rocblas", packages)
        self.assertIn("hipblas", packages)
        self.assertNotIn("hipblaslt", packages, "FAIL: hipblaslt is TRANSITIVE and should NOT be tested")

        # When hipblas changes: test hipblas and hipblaslt (DIRECT dependent only)
        # Do NOT test rocblas (upstream)
        packages = analyzer.get_packages_to_test(["hipblas"])
        self.assertIn("hipblas", packages)
        self.assertIn("hipblaslt", packages)
        self.assertNotIn("rocblas", packages, "FAIL: rocblas is UPSTREAM and should NOT be tested")

        # When hipblaslt changes: test only hipblaslt (no direct dependents)
        # Do NOT test hipblas or rocblas (both upstream)
        packages = analyzer.get_packages_to_test(["hipblaslt"])
        self.assertIn("hipblaslt", packages)
        self.assertNotIn("hipblas", packages, "FAIL: hipblas is UPSTREAM and should NOT be tested")
        self.assertNotIn("rocblas", packages, "FAIL: rocblas is UPSTREAM and should NOT be tested")


    def test_direct_dependencies_only_complex(self):
        """
        Test that we only test DIRECT dependencies, not transitive ones.

        Dependency chain: rocblas ← hipblas ← hipblaslt ← miopen ← miopenprovider
        """
        cmake_content = """
            therock_cmake_subproject_declare(rocBLAS
                RUNTIME_DEPS
                    hip-clr
            )

            therock_cmake_subproject_declare(hipBLAS
                RUNTIME_DEPS
                    rocBLAS
            )

            therock_cmake_subproject_declare(hipBLASLt
                RUNTIME_DEPS
                    hipBLAS
            )

            therock_cmake_subproject_declare(miopen
                RUNTIME_DEPS
                    hipBLASLt
            )

            therock_cmake_subproject_declare(miopenprovider
                RUNTIME_DEPS
                    miopen
            )
        """
        self.write_cmake_file(self.math_libs_dir / "CMakeLists.txt", cmake_content)
        # Use default (include_runtime_deps=True, include_build_deps=False)
        analyzer = PackageDependencyAnalyzer(self.therock_root)

        # When rocblas changes: only test hipblas (direct dependent)
        # Do NOT test hipblaslt, miopen, miopenprovider (all transitive)
        packages = analyzer.get_packages_to_test(["rocblas"])
        self.assertIn("rocblas", packages)
        self.assertIn("hipblas", packages)
        self.assertNotIn("hipblaslt", packages, "hipblaslt is transitive")
        self.assertNotIn("miopen", packages, "miopen is transitive")
        self.assertNotIn("miopenprovider", packages, "miopenprovider is transitive")

        # When hipblaslt changes: only test miopen (direct dependent)
        # Do NOT test miopenprovider (transitive) or hipblas/rocblas (upstream)
        packages = analyzer.get_packages_to_test(["hipblaslt"])
        self.assertIn("hipblaslt", packages)
        self.assertIn("miopen", packages)
        self.assertNotIn("miopenprovider", packages, "miopenprovider is transitive, not direct")
        self.assertNotIn("hipblas", packages, "hipblas is upstream")
        self.assertNotIn("rocblas", packages, "rocblas is upstream")


if __name__ == "__main__":
    unittest.main()
