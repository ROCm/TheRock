#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for source_path-to-artifact mapping in BuildTopology."""

import os
import re
import sys
import unittest
from pathlib import Path

BUILD_TOOLS_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, os.fspath(BUILD_TOOLS_DIR))

from _therock_utils.build_topology import BuildTopology, get_topology

REPO_ROOT = BUILD_TOOLS_DIR.parent


class ExtractSourcePathFromPathTest(unittest.TestCase):
    def test_projects_path(self):
        self.assertEqual(
            BuildTopology.extract_source_path_from_path("projects/rocblas/src/foo.cpp"),
            "rocblas",
        )

    def test_shared_path(self):
        self.assertEqual(
            BuildTopology.extract_source_path_from_path(
                "shared/rocroller/include/bar.hpp"
            ),
            "rocroller",
        )

    def test_dnn_providers_path(self):
        self.assertEqual(
            BuildTopology.extract_source_path_from_path(
                "dnn-providers/miopen-provider/src/baz.cpp"
            ),
            "miopen-provider",
        )

    def test_root_file_returns_none(self):
        self.assertIsNone(BuildTopology.extract_source_path_from_path("CMakeLists.txt"))


class GetArtifactForSourcePathTest(unittest.TestCase):
    def setUp(self):
        self.topology = get_topology()

    def test_rocblas_maps_to_blas(self):
        self.assertEqual(self.topology.get_artifact_for_source_path("rocblas"), "blas")

    def test_hipblas_maps_to_blas(self):
        self.assertEqual(self.topology.get_artifact_for_source_path("hipblas"), "blas")

    def test_rocrand_maps_to_rand(self):
        self.assertEqual(self.topology.get_artifact_for_source_path("rocrand"), "rand")

    def test_unknown_source_path_returns_none(self):
        self.assertIsNone(
            self.topology.get_artifact_for_source_path("unknown-source-path")
        )


class GetArtifactForPathTest(unittest.TestCase):
    def setUp(self):
        self.topology = get_topology()

    def test_rocblas_maps_to_blas(self):
        self.assertEqual(
            self.topology.get_artifact_for_path("projects/rocblas/src/foo.cpp"),
            "blas",
        )

    def test_rocfft_maps_to_fft(self):
        self.assertEqual(
            self.topology.get_artifact_for_path("projects/rocfft/src/kernel.cpp"),
            "fft",
        )

    def test_shared_rocroller_maps_to_blas(self):
        self.assertEqual(
            self.topology.get_artifact_for_path("shared/rocroller/src/foo.cpp"),
            "blas",
        )

    def test_unknown_path_returns_none(self):
        self.assertIsNone(self.topology.get_artifact_for_path("cmake/FindHIP.cmake"))


class ParseChangedPathTest(unittest.TestCase):
    def test_simple_path(self):
        submodule, subpath = BuildTopology.parse_changed_path(
            "rocm-libraries/projects/rocblas/foo.cpp"
        )
        self.assertEqual(submodule, "rocm-libraries")
        self.assertEqual(subpath, "projects/rocblas/foo.cpp")

    def test_single_component(self):
        submodule, subpath = BuildTopology.parse_changed_path("rocm-libraries")
        self.assertEqual(submodule, "rocm-libraries")
        self.assertEqual(subpath, "")


class SourceSetsWithSourcePathsTest(unittest.TestCase):
    def setUp(self):
        self.topology = get_topology()

    def test_returns_rocm_libraries(self):
        self.assertIn(
            "rocm-libraries", self.topology.get_source_sets_with_source_paths()
        )

    def test_returns_rocm_systems(self):
        self.assertIn("rocm-systems", self.topology.get_source_sets_with_source_paths())


class SourcePathsInSyncTest(unittest.TestCase):
    """Verify BUILD_TOPOLOGY.toml source_paths match CMakeLists.txt."""

    def test_cmake_source_paths_in_topology(self):
        topology = get_topology()
        topology_source_paths = set()
        for artifact in topology.artifacts.values():
            topology_source_paths.update(artifact.source_paths)

        cmake_source_paths = self._extract_cmake_source_paths()
        missing = cmake_source_paths - topology_source_paths
        self.assertEqual(
            missing,
            set(),
            f"Source paths in CMakeLists.txt but not BUILD_TOPOLOGY.toml: {sorted(missing)}",
        )

    def _extract_cmake_source_paths(self) -> set[str]:
        source_paths: set[str] = set()
        patterns = [
            re.compile(
                r'EXTERNAL_SOURCE_DIR\s+"?\$\{THEROCK_ROCM_LIBRARIES_SOURCE_DIR\}'
                r"/(?:projects|shared|dnn-providers)/([a-zA-Z0-9_-]+)"
            ),
            re.compile(
                r'EXTERNAL_SOURCE_DIR\s+"?\$\{THEROCK_ROCM_SYSTEMS_SOURCE_DIR\}'
                r"/(?:projects|shared)/([a-zA-Z0-9_-]+)"
            ),
        ]
        for cmake_file in REPO_ROOT.rglob("CMakeLists.txt"):
            rel_path = cmake_file.relative_to(REPO_ROOT)
            if any(
                part in ("rocm-libraries", "rocm-systems", "build", ".git")
                for part in rel_path.parts
            ):
                continue
            try:
                content = cmake_file.read_text()
            except Exception:
                continue
            for pattern in patterns:
                for match in pattern.finditer(content):
                    source_paths.add(match.group(1))
        return source_paths


if __name__ == "__main__":
    unittest.main()
