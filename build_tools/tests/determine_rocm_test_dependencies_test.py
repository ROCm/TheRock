# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import (
    ArtifactDependencyAnalyzer,
    create_analyzer,
)


class ArtifactDependencyAnalyzerTest(unittest.TestCase):
    """Tests for ArtifactDependencyAnalyzer using BUILD_TOPOLOGY.toml."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.therock_root = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def write_topology_file(self, content: str):
        """Write a BUILD_TOPOLOGY.toml file."""
        topology_file = self.therock_root / "BUILD_TOPOLOGY.toml"
        with open(topology_file, "w") as f:
            f.write(content)

    def test_basic_parsing(self):
        """Sanity check: can parse a basic BUILD_TOPOLOGY.toml."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.blas]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["core-hip"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        self.assertIn("blas", analyzer.artifacts)
        self.assertEqual(analyzer.artifacts["blas"].artifact_group, "math-libs")

    def test_get_packages_to_test_with_test_deps(self):
        """Test that test_deps field limits testing to specific packages."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.core-hip]
artifact_group = "hip-runtime"
type = "target-neutral"
artifact_deps = []

[artifacts.blas]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["core-hip"]
test_deps = ["solver"]

[artifacts.solver]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["blas"]

[artifacts.miopen]
artifact_group = "ml-libs"
type = "target-specific"
artifact_deps = ["blas"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # blas has test_deps = ["solver"], so only solver is tested (not miopen)
        packages = analyzer.get_packages_to_test(["blas"])
        self.assertIn("blas", packages)
        self.assertIn("solver", packages)
        # miopen should NOT be included due to test_deps override
        self.assertNotIn("miopen", packages)

    def test_get_packages_to_test_no_test_deps(self):
        """Test that packages without test_deps include all direct dependents."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.core-hip]
artifact_group = "hip-runtime"
type = "target-neutral"
artifact_deps = []

[artifacts.rand]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["core-hip"]

[artifacts.miopen]
artifact_group = "ml-libs"
type = "target-specific"
artifact_deps = ["rand"]

[artifacts.some-other]
artifact_group = "ml-libs"
type = "target-specific"
artifact_deps = ["rand"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # rand has no test_deps, so all direct dependents are included
        packages = analyzer.get_packages_to_test(["rand"])
        self.assertIn("rand", packages)
        self.assertIn("miopen", packages)
        self.assertIn("some-other", packages)

    def test_reverse_deps_built_correctly(self):
        """Test that reverse dependency graph is built correctly."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.a]
artifact_group = "test"
type = "target-neutral"
artifact_deps = []

[artifacts.b]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["a"]

[artifacts.c]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["a", "b"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # a is depended on by b and c
        self.assertEqual(analyzer.reverse_deps["a"], {"b", "c"})
        # b is depended on by c
        self.assertEqual(analyzer.reverse_deps["b"], {"c"})
        # c has no dependents
        self.assertEqual(analyzer.reverse_deps["c"], set())

    def test_create_analyzer(self):
        """Sanity check: create_analyzer helper works."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.blas]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = []
"""
        self.write_topology_file(toml_content)
        analyzer = create_analyzer(self.therock_root)

        self.assertIn("blas", analyzer.artifacts)

    def test_missing_topology_file(self):
        """Test error handling when BUILD_TOPOLOGY.toml is missing."""
        with self.assertRaises(FileNotFoundError):
            ArtifactDependencyAnalyzer(self.therock_root)


    def test_empty_test_deps(self):
        """Test that empty test_deps means only test the package itself."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.standalone]
artifact_group = "test"
type = "target-neutral"
artifact_deps = []
test_deps = []

[artifacts.dependent]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["standalone"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # standalone has test_deps = [], so only itself is tested
        packages = analyzer.get_packages_to_test(["standalone"])
        self.assertIn("standalone", packages)
        self.assertNotIn("dependent", packages)

    def test_test_deps_field_parsed(self):
        """Test that test_deps field is correctly parsed from TOML."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.pkg-with-override]
artifact_group = "test"
type = "target-neutral"
artifact_deps = []
test_deps = ["dep1", "dep2"]

[artifacts.pkg-without-override]
artifact_group = "test"
type = "target-neutral"
artifact_deps = []
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # Check that test_deps is parsed correctly
        self.assertIsNotNone(analyzer.artifacts["pkg-with-override"].test_deps)
        self.assertEqual(
            analyzer.artifacts["pkg-with-override"].test_deps, {"dep1", "dep2"}
        )

        # Check that packages without test_deps have None
        self.assertIsNone(analyzer.artifacts["pkg-without-override"].test_deps)

    def test_multiple_changed_packages(self):
        """Test handling multiple changed packages with mixed override settings."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.base]
artifact_group = "test"
type = "target-neutral"
artifact_deps = []

[artifacts.pkg-a]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["base"]
test_deps = ["specific-dep"]

[artifacts.pkg-b]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["base"]

[artifacts.specific-dep]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["pkg-a"]

[artifacts.dep-of-pkg-b]
artifact_group = "test"
type = "target-neutral"
artifact_deps = ["pkg-b"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # Test both packages changed: pkg-a uses test_deps, pkg-b uses reverse deps
        packages = analyzer.get_packages_to_test(["pkg-a", "pkg-b"])
        self.assertIn("pkg-a", packages)
        self.assertIn("specific-dep", packages)  # From pkg-a's test_deps
        self.assertIn("pkg-b", packages)
        self.assertIn("dep-of-pkg-b", packages)  # From pkg-b's reverse deps

    def test_real_world_blas_scenario(self):
        """Test realistic scenario similar to blas -> solver."""
        toml_content = """
[metadata]
version = "2.0"

[artifacts.blas]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = []
test_deps = ["solver"]

[artifacts.solver]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["blas"]

[artifacts.sparse]
artifact_group = "math-libs"
type = "target-specific"
artifact_deps = ["blas"]

[artifacts.miopen]
artifact_group = "ml-libs"
type = "target-specific"
artifact_deps = ["blas"]
"""
        self.write_topology_file(toml_content)
        analyzer = ArtifactDependencyAnalyzer(self.therock_root)

        # When blas changes, only test blas and solver (not sparse or miopen)
        packages = analyzer.get_packages_to_test(["blas"])
        self.assertEqual(packages, {"blas", "solver"})


if __name__ == "__main__":
    unittest.main()
