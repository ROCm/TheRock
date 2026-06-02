#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for configure_downstream_tests module.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "github_actions"))

from _therock_utils.build_topology import BuildTopology

from configure_downstream_tests import (
    ARTIFACT_TO_TEST_LABELS,
    _is_infrastructure_file,
    _parse_comma_list,
    _parse_gitmodules,
    _build_submodule_to_source_sets,
    _build_source_set_to_artifact_groups,
    build_per_family_info,
    detect_changed_artifacts_from_files,
    get_downstream_test_labels,
)

# A minimal topology for unit testing with realistic dependency patterns.
SAMPLE_TOPOLOGY = textwrap.dedent("""\
    [metadata]
    version = "1.0"

    [source_sets.core-src]
    description = "Core submodules"
    submodules = ["rocm-systems"]

    [source_sets.math-src]
    description = "Math submodules"
    submodules = ["rocm-libraries"]

    [source_sets.compiler-src]
    description = "Compiler submodules"
    submodules = ["llvm-project"]

    [artifact_groups.core]
    description = "Core group"
    type = "generic"
    source_sets = ["core-src"]

    [artifact_groups.math]
    description = "Math group"
    type = "per-arch"
    artifact_group_deps = ["core"]
    source_sets = ["math-src"]

    [artifact_groups.ml]
    description = "ML group"
    type = "per-arch"
    artifact_group_deps = ["math"]
    source_sets = ["math-src"]

    [artifact_groups.comm]
    description = "Comm group"
    type = "generic"
    artifact_group_deps = ["core"]
    source_sets = ["core-src"]

    [artifact_groups.profiler]
    description = "Profiler group"
    type = "generic"
    artifact_group_deps = ["core"]
    source_sets = ["core-src"]

    [artifact_groups.compiler]
    description = "Compiler group"
    type = "generic"
    source_sets = ["compiler-src"]

    [artifact_groups.misc]
    description = "Misc group"
    type = "generic"
    source_sets = []

    [artifacts.core-hip]
    artifact_group = "core"
    type = "target-specific"

    [artifacts.core-runtime]
    artifact_group = "core"
    type = "target-specific"

    [artifacts.blas]
    artifact_group = "math"
    type = "target-specific"
    artifact_deps = ["core-hip"]

    [artifacts.sparse]
    artifact_group = "math"
    type = "target-specific"
    artifact_deps = ["blas"]

    [artifacts.solver]
    artifact_group = "math"
    type = "target-specific"
    artifact_deps = ["blas", "sparse"]

    [artifacts.prim]
    artifact_group = "math"
    type = "target-specific"
    artifact_deps = ["core-hip"]

    [artifacts.rand]
    artifact_group = "math"
    type = "target-specific"
    artifact_deps = ["core-hip"]

    [artifacts.miopen]
    artifact_group = "ml"
    type = "target-specific"
    artifact_deps = ["blas"]

    [artifacts.rccl]
    artifact_group = "comm"
    type = "target-specific"
    artifact_deps = ["core-hip"]

    [artifacts.rocprofiler-sdk]
    artifact_group = "profiler"
    type = "target-specific"
    artifact_deps = ["core-runtime"]

    [artifacts.amd-llvm]
    artifact_group = "compiler"
    type = "target-neutral"

    [artifacts.leaf-no-tests]
    artifact_group = "misc"
    type = "target-neutral"
""")

# Matching .gitmodules content for the sample topology.
SAMPLE_GITMODULES = textwrap.dedent("""\
    [submodule "rocm-systems"]
    \tpath = rocm-systems
    \turl = https://github.com/ROCm/rocm-systems.git
    [submodule "rocm-libraries"]
    \tpath = rocm-libraries
    \turl = https://github.com/ROCm/rocm-libraries.git
    [submodule "llvm-project"]
    \tpath = compiler/amd-llvm
    \turl = https://github.com/ROCm/llvm-project.git
""")


class TestParseCommaList(unittest.TestCase):
    """Test _parse_comma_list utility."""

    def test_simple(self):
        self.assertEqual(_parse_comma_list("blas,prim"), ["blas", "prim"])

    def test_whitespace(self):
        self.assertEqual(
            _parse_comma_list(" blas , prim , rand "), ["blas", "prim", "rand"]
        )

    def test_empty(self):
        self.assertEqual(_parse_comma_list(""), [])

    def test_single(self):
        self.assertEqual(_parse_comma_list("blas"), ["blas"])

    def test_trailing_comma(self):
        self.assertEqual(_parse_comma_list("blas,"), ["blas"])


class TestArtifactToTestLabels(unittest.TestCase):
    """Test that the mapping dict has expected structure."""

    def test_blas_has_labels(self):
        labels = ARTIFACT_TO_TEST_LABELS["blas"]
        self.assertIn("rocblas", labels)
        self.assertIn("hipblas", labels)

    def test_prim_has_labels(self):
        labels = ARTIFACT_TO_TEST_LABELS["prim"]
        self.assertIn("rocprim", labels)
        self.assertIn("hipcub", labels)
        self.assertIn("rocthrust", labels)

    def test_core_hip_has_labels(self):
        labels = ARTIFACT_TO_TEST_LABELS["core-hip"]
        self.assertIn("hip-tests", labels)

    def test_empty_labels_for_composable_kernel(self):
        self.assertEqual(ARTIFACT_TO_TEST_LABELS["composable-kernel"], [])

    def test_all_values_are_lists(self):
        for key, value in ARTIFACT_TO_TEST_LABELS.items():
            self.assertIsInstance(value, list, f"Value for {key!r} is not a list")


class TestGetDownstreamTestLabels(unittest.TestCase):
    """Test downstream test label computation with a synthetic topology."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        )
        self.temp_file.write(SAMPLE_TOPOLOGY)
        self.temp_file.close()
        self.topology = BuildTopology(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_blas_downstream(self):
        """Changing blas should also test sparse, solver, and miopen."""
        labels, affected = get_downstream_test_labels(["blas"], self.topology)
        # blas itself
        self.assertIn("blas", affected)
        # Downstream of blas
        self.assertIn("sparse", affected)
        self.assertIn("solver", affected)
        self.assertIn("miopen", affected)
        # Labels should include both blas's own tests and downstream tests
        self.assertIn("test:rocblas", labels)
        self.assertIn("test:hipsparse", labels)
        self.assertIn("test:rocsparse", labels)
        self.assertIn("test:miopen", labels)

    def test_prim_no_downstream(self):
        """prim is a leaf in the sample topology, so only its own labels."""
        labels, affected = get_downstream_test_labels(["prim"], self.topology)
        self.assertEqual(affected, {"prim"})
        self.assertIn("test:rocprim", labels)
        self.assertIn("test:hipcub", labels)
        self.assertIn("test:rocthrust", labels)

    def test_core_hip_downstream(self):
        """Changing core-hip should cascade to blas, sparse, solver, etc."""
        labels, affected = get_downstream_test_labels(["core-hip"], self.topology)
        self.assertIn("core-hip", affected)
        self.assertIn("blas", affected)
        self.assertIn("sparse", affected)
        self.assertIn("solver", affected)
        self.assertIn("prim", affected)
        self.assertIn("rand", affected)
        self.assertIn("miopen", affected)
        self.assertIn("rccl", affected)
        # Should have hip-tests from core-hip
        self.assertIn("test:hip-tests", labels)
        # And downstream tests
        self.assertIn("test:rocblas", labels)

    def test_multiple_changed_artifacts(self):
        """Union of downstream sets from multiple changed artifacts."""
        labels, affected = get_downstream_test_labels(
            ["prim", "rand"], self.topology
        )
        self.assertIn("prim", affected)
        self.assertIn("rand", affected)
        self.assertIn("test:rocprim", labels)
        self.assertIn("test:rocrand", labels)

    def test_unknown_artifact(self):
        """Unknown artifacts should be skipped with a warning, not raise."""
        labels, affected = get_downstream_test_labels(
            ["nonexistent"], self.topology
        )
        # The unknown artifact is still in the affected set
        self.assertIn("nonexistent", affected)
        # But it has no test labels since it's not in the mapping
        self.assertEqual(labels, [])

    def test_empty_changed_artifacts(self):
        """Empty input should produce empty output."""
        labels, affected = get_downstream_test_labels([], self.topology)
        self.assertEqual(labels, [])
        self.assertEqual(affected, set())

    def test_artifact_with_no_test_mapping(self):
        """An artifact in the topology but not in the label mapping."""
        labels, affected = get_downstream_test_labels(
            ["leaf-no-tests"], self.topology
        )
        self.assertIn("leaf-no-tests", affected)
        self.assertEqual(labels, [])

    def test_labels_are_sorted(self):
        """Test labels should be returned in sorted order."""
        labels, _ = get_downstream_test_labels(["blas"], self.topology)
        self.assertEqual(labels, sorted(labels))

    def test_labels_have_test_prefix(self):
        """All returned labels should have the 'test:' prefix."""
        labels, _ = get_downstream_test_labels(["blas"], self.topology)
        for label in labels:
            self.assertTrue(
                label.startswith("test:"), f"Label {label!r} missing 'test:' prefix"
            )


class TestBuildPerFamilyInfo(unittest.TestCase):
    """Test per-family configuration building."""

    def test_with_mock_families(self):
        """Test per_family_info generation from a mock family dict."""
        mock_families = {
            "gfx94x": {
                "linux": {
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                    "test-runs-on": "linux-gfx942-1gpu-ossci-rocm",
                    "build_variants": ["release"],
                },
            },
            "gfx110x": {
                "linux": {
                    "family": "gfx110X-all",
                    "fetch-gfx-targets": ["gfx1100", "gfx1101"],
                    "test-runs-on": "linux-gfx1100-1gpu-ossci-rocm",
                    "build_variants": ["release"],
                },
            },
        }

        result = build_per_family_info("linux", families=mock_families)
        self.assertEqual(len(result), 2)

        families_in_result = {r["amdgpu_family"] for r in result}
        self.assertIn("gfx94X-dcgpu", families_in_result)
        self.assertIn("gfx110X-all", families_in_result)

        for entry in result:
            self.assertIn("amdgpu_family", entry)
            self.assertIn("amdgpu_targets", entry)
            self.assertIn("test-runs-on", entry)
            self.assertIn("sanity_check_only_for_family", entry)

    def test_skips_families_without_platform(self):
        """Families without the requested platform should be skipped."""
        mock_families = {
            "gfx94x": {
                "linux": {
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                    "test-runs-on": "some-runner",
                    "build_variants": ["release"],
                },
            },
        }
        result = build_per_family_info("windows", families=mock_families)
        self.assertEqual(len(result), 0)

    def test_skips_families_without_test_runner(self):
        """Families without a test runner should be skipped."""
        mock_families = {
            "gfx94x": {
                "linux": {
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                    "test-runs-on": "",
                    "build_variants": ["release"],
                },
            },
        }
        result = build_per_family_info("linux", families=mock_families)
        self.assertEqual(len(result), 0)

    def test_multiple_gfx_targets_joined(self):
        """Multiple gfx targets should be comma-joined."""
        mock_families = {
            "gfx110x": {
                "linux": {
                    "family": "gfx110X-all",
                    "fetch-gfx-targets": ["gfx1100", "gfx1101"],
                    "test-runs-on": "some-runner",
                    "build_variants": ["release"],
                },
            },
        }
        result = build_per_family_info("linux", families=mock_families)
        self.assertEqual(result[0]["amdgpu_targets"], "gfx1100,gfx1101")

    def test_sanity_check_only_propagated(self):
        """sanity_check_only_for_family should be propagated from family config."""
        mock_families = {
            "gfx94x": {
                "linux": {
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                    "test-runs-on": "some-runner",
                    "build_variants": ["release"],
                    "sanity_check_only_for_family": True,
                },
            },
        }
        result = build_per_family_info("linux", families=mock_families)
        self.assertTrue(result[0]["sanity_check_only_for_family"])


class TestParseGitmodules(unittest.TestCase):
    """Test .gitmodules parsing."""

    def test_parse_sample(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gitmodules", delete=False
        ) as f:
            f.write(SAMPLE_GITMODULES)
            f.flush()
            result = _parse_gitmodules(Path(f.name))
        os.unlink(f.name)

        self.assertEqual(result["rocm-systems"], "rocm-systems")
        self.assertEqual(result["rocm-libraries"], "rocm-libraries")
        self.assertEqual(result["compiler/amd-llvm"], "llvm-project")

    def test_parse_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gitmodules", delete=False
        ) as f:
            f.write("")
            f.flush()
            result = _parse_gitmodules(Path(f.name))
        os.unlink(f.name)
        self.assertEqual(result, {})


class TestDetectChangedArtifactsFromFiles(unittest.TestCase):
    """Test auto-detection of changed artifacts from file paths."""

    def setUp(self):
        # Write topology
        self.topo_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        )
        self.topo_file.write(SAMPLE_TOPOLOGY)
        self.topo_file.close()
        self.topology = BuildTopology(self.topo_file.name)

        # Write gitmodules
        self.gitmod_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".gitmodules", delete=False
        )
        self.gitmod_file.write(SAMPLE_GITMODULES)
        self.gitmod_file.close()
        self.gitmodules_path = Path(self.gitmod_file.name)

    def tearDown(self):
        os.unlink(self.topo_file.name)
        os.unlink(self.gitmod_file.name)

    def test_submodule_path_maps_to_artifacts(self):
        """A change in rocm-libraries maps to math and ml artifacts."""
        result, infra = detect_changed_artifacts_from_files(
            ["rocm-libraries"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        # rocm-libraries -> source_set "math-src" -> artifact_groups "math", "ml"
        self.assertIn("blas", result)
        self.assertIn("prim", result)
        self.assertIn("miopen", result)
        self.assertFalse(infra)

    def test_submodule_subpath_maps_to_artifacts(self):
        """A file within a submodule path should still match."""
        result, infra = detect_changed_artifacts_from_files(
            ["compiler/amd-llvm/lib/Target/AMDGPU/foo.cpp"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        # compiler/amd-llvm -> llvm-project -> compiler-src -> compiler group
        self.assertIn("amd-llvm", result)
        self.assertFalse(infra)

    def test_rocm_systems_maps_to_core(self):
        """rocm-systems submodule maps to core artifacts."""
        result, infra = detect_changed_artifacts_from_files(
            ["rocm-systems"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        self.assertIn("core-hip", result)
        self.assertIn("core-runtime", result)
        self.assertFalse(infra)

    def test_unmapped_files_return_empty(self):
        """Infrastructure files not under any submodule flag infra_changed."""
        result, infra = detect_changed_artifacts_from_files(
            ["build_tools/some_script.py", "cmake/foo.cmake"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        self.assertEqual(result, [])
        self.assertTrue(infra)

    def test_doc_only_files_no_infra_flag(self):
        """Doc-only changes should not flag infrastructure_changed."""
        result, infra = detect_changed_artifacts_from_files(
            ["README.md", "docs/guide.md", "LICENSE"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        self.assertEqual(result, [])
        self.assertFalse(infra)

    def test_mixed_files(self):
        """Mix of submodule and non-submodule files."""
        result, infra = detect_changed_artifacts_from_files(
            ["rocm-libraries", "build_tools/foo.py", "README.md"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        self.assertIn("blas", result)
        # build_tools is infrastructure
        self.assertTrue(infra)

    def test_empty_file_list(self):
        """Empty input returns empty output."""
        result, infra = detect_changed_artifacts_from_files(
            [],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        self.assertEqual(result, [])
        self.assertFalse(infra)

    def test_multiple_submodules_union(self):
        """Changes in multiple submodules produce the union of artifacts."""
        result, infra = detect_changed_artifacts_from_files(
            ["rocm-systems", "compiler/amd-llvm"],
            topology=self.topology,
            gitmodules_path=self.gitmodules_path,
        )
        # Should have core artifacts + compiler artifacts
        self.assertIn("core-hip", result)
        self.assertIn("amd-llvm", result)
        self.assertFalse(infra)


class TestIsInfrastructureFile(unittest.TestCase):
    """Test infrastructure file classification."""

    def test_cmake_files(self):
        self.assertTrue(_is_infrastructure_file("cmake/foo.cmake"))
        self.assertTrue(_is_infrastructure_file("cmake/therock/FindROCm.cmake"))

    def test_build_tools(self):
        self.assertTrue(_is_infrastructure_file("build_tools/some_script.py"))
        self.assertTrue(
            _is_infrastructure_file("build_tools/github_actions/configure_ci.py")
        )

    def test_patches(self):
        self.assertTrue(_is_infrastructure_file("patches/llvm-project/foo.patch"))

    def test_top_level_infra_files(self):
        self.assertTrue(_is_infrastructure_file("CMakeLists.txt"))
        self.assertTrue(_is_infrastructure_file("FLAGS.cmake"))
        self.assertTrue(_is_infrastructure_file("BUILD_TOPOLOGY.toml"))

    def test_component_wrapper_files(self):
        self.assertTrue(_is_infrastructure_file("core/CMakeLists.txt"))
        self.assertTrue(_is_infrastructure_file("math-libs/pre_hook_blas.cmake"))
        self.assertTrue(_is_infrastructure_file("ml-libs/artifact-miopen.toml"))

    def test_docs_not_infra(self):
        self.assertFalse(_is_infrastructure_file("README.md"))
        self.assertFalse(_is_infrastructure_file("docs/guide.md"))
        self.assertFalse(_is_infrastructure_file("CONTRIBUTING.md"))
        self.assertFalse(_is_infrastructure_file("LICENSE"))

    def test_github_workflows(self):
        self.assertTrue(
            _is_infrastructure_file(".github/workflows/multi_arch_ci.yml")
        )

    def test_version_json_not_infra(self):
        """version.json is not in the infra list (low risk)."""
        self.assertFalse(_is_infrastructure_file("version.json"))


if __name__ == "__main__":
    unittest.main()
