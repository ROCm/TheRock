# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import (
    DIRECT_DEPENDENT_OVERRIDES,
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

    def test_get_packages_to_test_with_deps(self):
        """Sanity check: get_packages_to_test returns expected packages."""
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

        # blas has override to only test solver
        packages = analyzer.get_packages_to_test(["blas"])
        self.assertIn("blas", packages)
        self.assertIn("solver", packages)
        # miopen should NOT be included due to override
        self.assertNotIn("miopen", packages)

    def test_get_packages_to_test_no_override(self):
        """Test that packages without overrides include all direct dependents."""
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

        # rand has no override, so all direct dependents are included
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


class DirectDependentOverridesTest(unittest.TestCase):
    """Tests for the DIRECT_DEPENDENT_OVERRIDES configuration."""

    def test_blas_override_exists(self):
        """Verify blas override is configured."""
        self.assertIn("blas", DIRECT_DEPENDENT_OVERRIDES)
        self.assertIn("solver", DIRECT_DEPENDENT_OVERRIDES["blas"])


if __name__ == "__main__":
    unittest.main()
