# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import (
    SubprojectDependencyAnalyzer,
    create_analyzer,
    get_rocm_test_dependencies,
)


class SubprojectDependencyAnalyzerTest(unittest.TestCase):
    """Tests for SubprojectDependencyAnalyzer using CMake manifest."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.build_dir = Path(self.temp_dir) / "build"
        self.build_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def write_manifest(self, manifest_data: dict):
        """Write a subproject_test_manifest.json file."""
        manifest_file = self.build_dir / "subproject_test_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f, indent=2)

    def test_basic_parsing(self):
        """Sanity check: can parse a basic manifest."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "rocBLAS": {
                    "runtime_deps": ["hip-clr"],
                }
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        self.assertIn("rocBLAS", analyzer.subprojects)
        self.assertEqual(analyzer.subprojects["rocBLAS"].runtime_deps, {"hip-clr"})

    def test_get_subprojects_to_test_with_test_subprojects(self):
        """Test that test_subprojects field limits testing to specific subprojects."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "hip-clr": {"runtime_deps": []},
                "rocBLAS": {
                    "runtime_deps": ["hip-clr"],
                    "test_subprojects": ["rocSOLVER", "hipBLAS"],
                },
                "rocSOLVER": {"runtime_deps": ["rocBLAS"]},
                "hipBLAS": {"runtime_deps": ["rocBLAS"]},
                "MIOpen": {"runtime_deps": ["rocBLAS"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # rocBLAS has test_subprojects = ["rocSOLVER", "hipBLAS"], so only those are tested (not MIOpen)
        subprojects = analyzer.get_subprojects_to_test(["rocBLAS"])
        self.assertIn("rocBLAS", subprojects)
        self.assertIn("rocSOLVER", subprojects)
        self.assertIn("hipBLAS", subprojects)
        # MIOpen should NOT be included due to test_subprojects override
        self.assertNotIn("MIOpen", subprojects)

    def test_get_subprojects_to_test_no_test_subprojects(self):
        """Test that subprojects without test_subprojects include all direct dependents."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "hip-clr": {"runtime_deps": []},
                "rocRAND": {"runtime_deps": ["hip-clr"]},
                "MIOpen": {"runtime_deps": ["rocRAND"]},
                "some-other": {"runtime_deps": ["rocRAND"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # rocRAND has no test_subprojects, so all direct dependents are included
        subprojects = analyzer.get_subprojects_to_test(["rocRAND"])
        self.assertIn("rocRAND", subprojects)
        self.assertIn("MIOpen", subprojects)
        self.assertIn("some-other", subprojects)

    def test_reverse_deps_built_correctly(self):
        """Test that reverse dependency graph is built correctly."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "a": {"runtime_deps": []},
                "b": {"runtime_deps": ["a"]},
                "c": {"runtime_deps": ["a", "b"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # a is depended on by b and c
        self.assertEqual(analyzer.reverse_deps["a"], {"b", "c"})
        # b is depended on by c
        self.assertEqual(analyzer.reverse_deps["b"], {"c"})
        # c has no dependents
        self.assertEqual(analyzer.reverse_deps["c"], set())

    def test_create_analyzer(self):
        """Sanity check: create_analyzer helper works."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "rocBLAS": {"runtime_deps": []},
            },
        }
        self.write_manifest(manifest)
        analyzer = create_analyzer(Path(self.temp_dir), self.build_dir)

        self.assertIn("rocBLAS", analyzer.subprojects)

    def test_missing_manifest_file(self):
        """Test error handling when manifest is missing."""
        with self.assertRaises(FileNotFoundError):
            SubprojectDependencyAnalyzer(
                self.build_dir / "subproject_test_manifest.json"
            )

    def test_empty_test_subprojects(self):
        """Test that empty test_subprojects means only test the subproject itself."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "standalone": {"runtime_deps": [], "test_subprojects": []},
                "dependent": {"runtime_deps": ["standalone"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # standalone has test_subprojects = [], so only itself is tested
        subprojects = analyzer.get_subprojects_to_test(["standalone"])
        self.assertIn("standalone", subprojects)
        self.assertNotIn("dependent", subprojects)

    def test_test_subprojects_field_parsed(self):
        """Test that test_subprojects field is correctly parsed from manifest."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "pkg-with-override": {
                    "runtime_deps": [],
                    "test_subprojects": ["dep1", "dep2"],
                },
                "pkg-without-override": {"runtime_deps": []},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # Check that test_subprojects is parsed correctly
        self.assertIsNotNone(
            analyzer.subprojects["pkg-with-override"].test_subprojects
        )
        self.assertEqual(
            analyzer.subprojects["pkg-with-override"].test_subprojects,
            {"dep1", "dep2"},
        )

        # Check that subprojects without test_subprojects have None
        self.assertIsNone(
            analyzer.subprojects["pkg-without-override"].test_subprojects
        )

    def test_multiple_changed_subprojects(self):
        """Test handling multiple changed subprojects with mixed override settings."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "base": {"runtime_deps": []},
                "pkg-a": {
                    "runtime_deps": ["base"],
                    "test_subprojects": ["specific-dep"],
                },
                "pkg-b": {"runtime_deps": ["base"]},
                "specific-dep": {"runtime_deps": ["pkg-a"]},
                "dep-of-pkg-b": {"runtime_deps": ["pkg-b"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # Test both subprojects changed: pkg-a uses test_subprojects, pkg-b uses reverse deps
        subprojects = analyzer.get_subprojects_to_test(["pkg-a", "pkg-b"])
        self.assertIn("pkg-a", subprojects)
        self.assertIn("specific-dep", subprojects)  # From pkg-a's test_subprojects
        self.assertIn("pkg-b", subprojects)
        self.assertIn("dep-of-pkg-b", subprojects)  # From pkg-b's reverse deps

    def test_real_world_rocblas_scenario(self):
        """Test realistic scenario similar to rocBLAS -> hipBLAS, rocSOLVER."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "rocBLAS": {
                    "runtime_deps": [],
                    "test_subprojects": ["hipBLAS", "rocSOLVER"],
                },
                "hipBLAS": {"runtime_deps": ["rocBLAS"]},
                "rocSOLVER": {"runtime_deps": ["rocBLAS"]},
                "rocSPARSE": {"runtime_deps": ["rocBLAS"]},
                "MIOpen": {"runtime_deps": ["rocBLAS"]},
            },
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        # When rocBLAS changes, only test rocBLAS + hipBLAS + rocSOLVER (not rocSPARSE or MIOpen)
        subprojects = analyzer.get_subprojects_to_test(["rocBLAS"])
        self.assertEqual(subprojects, {"rocBLAS", "hipBLAS", "rocSOLVER"})

    def test_get_rocm_test_dependencies_convenience_function(self):
        """Test the convenience function get_rocm_test_dependencies."""
        manifest = {
            "metadata": {"description": "Test"},
            "subprojects": {
                "rocBLAS": {
                    "runtime_deps": [],
                    "test_subprojects": ["hipBLAS", "rocSOLVER"],
                },
                "hipBLAS": {"runtime_deps": ["rocBLAS"]},
                "rocSOLVER": {"runtime_deps": ["rocBLAS"]},
            },
        }
        self.write_manifest(manifest)

        # Test the convenience function
        result = get_rocm_test_dependencies(
            changed_subprojects=["rocBLAS"],
            therock_dir=Path(self.temp_dir),
            build_dir=self.build_dir,
        )
        self.assertEqual(result, {"rocBLAS", "hipBLAS", "rocSOLVER"})


if __name__ == "__main__":
    unittest.main()
