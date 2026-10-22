#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for stage impact analysis."""

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

# build_tools/github_actions/tests -> build_tools
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from _therock_utils.build_topology import BuildTopology, get_topology
from github_actions.stage_impact import analyze_stage_impact


class StageImpactTest(unittest.TestCase):
    """Test cases for stage impact analysis."""

    def setUp(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as temp_file:
            self.topology_path = temp_file.name

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.topology_path):
            os.unlink(self.topology_path)

    def write_topology(self, content: str) -> None:
        """Write topology content to temp file."""
        with open(self.topology_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(content))

    def test_rocm_libraries_maps_to_math_libs(self):
        """A rocm-libraries submodule change should impact math-libs."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["rocm-libraries"], topology=topology)

        self.assertEqual(result.changed_inputs, ("rocm-libraries",))
        self.assertIn("rocm-libraries", result.matched_source_sets)
        self.assertIn("math-libs", result.impacted_artifact_groups)
        self.assertIn("math-libs", result.rebuild_stages)
        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(result.unmatched_inputs, ())

    def test_llvm_project_maps_to_compiler_runtime(self):
        """An llvm-project submodule change should impact compiler-runtime."""
        self.write_topology(
            """
            [source_sets.compilers]
            description = "Compiler toolchain submodules"
            submodules = ["llvm-project", "HIPIFY", "spirv-llvm-translator"]

            [artifact_groups.compiler]
            description = "Compiler"
            type = "generic"
            source_sets = ["compilers"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["compiler"]

            [artifacts.amd-llvm]
            artifact_group = "compiler"
            type = "target-neutral"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["llvm-project"], topology=topology)

        self.assertEqual(result.changed_inputs, ("llvm-project",))
        self.assertIn("compilers", result.matched_source_sets)
        self.assertIn("compiler", result.impacted_artifact_groups)
        self.assertIn("compiler-runtime", result.rebuild_stages)
        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(result.unmatched_inputs, ())

    def write_narrow_topology(self) -> None:
        """Topology with several independent stages for narrow-impact tests."""
        self.write_topology(
            """
            [source_sets.compilers]
            description = "Compiler toolchain submodules"
            submodules = ["llvm-project"]

            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [source_sets.amd-mesa]
            description = "ROCm media submodules"
            submodules = ["amd-mesa"]
            disable_platforms = ["windows"]

            [source_sets.debug-tools]
            description = "ROCm debug tools"
            submodules = ["rocgdb"]

            [artifact_groups.compiler]
            description = "Compiler"
            type = "generic"
            source_sets = ["compilers"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [artifact_groups.media-libs]
            description = "Media libs"
            type = "generic"
            source_sets = ["amd-mesa"]

            [artifact_groups.debug-tools]
            description = "Debug tools"
            type = "generic"
            source_sets = ["debug-tools"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["compiler"]

            [build_stages.math-libs]
            description = "Math libs"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [build_stages.media-libs]
            description = "Media libs"
            artifact_groups = ["media-libs"]

            [build_stages.debug-tools]
            description = "Debug tools"
            artifact_groups = ["debug-tools"]

            [artifacts.amd-llvm]
            artifact_group = "compiler"
            type = "target-neutral"

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"

            [artifacts.rocdecode]
            artifact_group = "media-libs"
            type = "target-neutral"

            [artifacts.rocgdb]
            artifact_group = "debug-tools"
            type = "target-neutral"
            """
        )

    def test_rocm_libraries_is_narrow(self):
        """rocm-libraries should only rebuild math-libs, not every stage."""
        self.write_narrow_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["rocm-libraries"], topology=topology)

        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(result.matched_source_sets, ("rocm-libraries",))
        self.assertEqual(result.impacted_artifact_groups, ("math-libs",))
        self.assertEqual(result.rebuild_stages, ("math-libs",))
        self.assertEqual(
            result.copy_stages,
            ("compiler-runtime", "debug-tools", "media-libs"),
        )
        self.assertEqual(result.unmatched_inputs, ())

    def test_amd_mesa_is_narrow(self):
        """amd-mesa should only rebuild media-libs, not every stage."""
        self.write_narrow_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["amd-mesa"], topology=topology, platform="linux")

        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(result.matched_source_sets, ("amd-mesa",))
        self.assertEqual(result.impacted_artifact_groups, ("media-libs",))
        self.assertEqual(result.rebuild_stages, ("media-libs",))
        self.assertEqual(
            result.copy_stages,
            ("compiler-runtime", "debug-tools", "math-libs"),
        )
        self.assertEqual(result.unmatched_inputs, ())

    def test_rocgdb_is_narrow(self):
        """rocgdb should only rebuild debug-tools, not every stage."""
        self.write_narrow_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["rocgdb"], topology=topology)

        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(result.matched_source_sets, ("debug-tools",))
        self.assertEqual(result.impacted_artifact_groups, ("debug-tools",))
        self.assertEqual(result.rebuild_stages, ("debug-tools",))
        self.assertEqual(
            result.copy_stages,
            ("compiler-runtime", "math-libs", "media-libs"),
        )
        self.assertEqual(result.unmatched_inputs, ())

    def test_unknown_input_forces_full_fallback(self):
        """Unknown input should force a conservative full-CI fallback."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["some-random-dir"], topology=topology)

        self.assertEqual(result.changed_inputs, ("some-random-dir",))
        self.assertTrue(result.full_rebuild_required)
        self.assertIn("some-random-dir", result.unmatched_inputs)
        self.assertGreaterEqual(len(result.rebuild_stages), 0)

    def test_topology_file_change_forces_full_fallback(self):
        """A BUILD_TOPOLOGY.toml change should force full CI fallback."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["BUILD_TOPOLOGY.toml"], topology=topology)

        self.assertEqual(result.changed_inputs, ("BUILD_TOPOLOGY.toml",))
        self.assertTrue(result.full_rebuild_required)
        self.assertIn("BUILD_TOPOLOGY.toml", result.reasons[0])

    def test_build_tools_change_forces_full_fallback(self):
        """A build_tools change should force full CI fallback."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["build_tools/github_actions/foo.py"], topology=topology
        )

        self.assertEqual(result.changed_inputs, ("build_tools/github_actions/foo.py",))
        self.assertTrue(result.full_rebuild_required)
        self.assertIn("build_tools/github_actions/foo.py", result.reasons[0])

    def test_multiple_inputs_deduplicate_impacted_stages(self):
        """Multiple changed inputs should union their impacted stages without duplicates."""
        self.write_topology(
            """
            [source_sets.compilers]
            description = "Compiler toolchain submodules"
            submodules = ["llvm-project"]

            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.compiler]
            description = "Compiler"
            type = "generic"
            source_sets = ["compilers"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["compiler"]

            [build_stages.math-libs]
            description = "Math libs"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.amd-llvm]
            artifact_group = "compiler"
            type = "target-neutral"

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries", "llvm-project"], topology=topology
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertEqual(
            set(result.matched_source_sets), {"compilers", "rocm-libraries"}
        )
        self.assertEqual(set(result.rebuild_stages), {"compiler-runtime", "math-libs"})
        self.assertEqual(len(result.rebuild_stages), 2)

    def test_platform_disabled_source_set_is_ignored(self):
        """Source sets disabled for the target platform should not contribute."""
        self.write_topology(
            """
            [source_sets.common]
            description = "Common"
            submodules = ["common"]

            [source_sets.windows-only]
            description = "Windows-only"
            submodules = ["windows-only"]
            disable_platforms = ["linux"]

            [artifact_groups.runtime]
            description = "Runtime"
            type = "generic"
            source_sets = ["common", "windows-only"]

            [build_stages.runtime]
            description = "Runtime"
            artifact_groups = ["runtime"]

            [artifacts.runtime-artifact]
            artifact_group = "runtime"
            type = "target-neutral"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["windows-only"], topology=topology, platform="linux"
        )

        self.assertTrue(result.full_rebuild_required)
        self.assertEqual(result.matched_source_sets, ())
        self.assertIn("windows-only", result.unmatched_inputs)

    def write_nested_path_topology(self) -> None:
        self.write_topology(
            """
            [source_sets.compilers]
            description = "Compiler toolchain submodules"
            submodules = ["llvm-project", "HIPIFY"]
            path_prefixes = ["compiler/amd-llvm", "compiler/hipify"]

            [source_sets.math-libs]
            description = "Math libraries"
            submodules = ["libhipcxx"]
            path_prefixes = ["math-libs/libhipcxx"]

            [artifact_groups.compiler]
            description = "Compiler"
            type = "generic"
            source_sets = ["compilers"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["math-libs"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["compiler"]

            [build_stages.math-libs]
            description = "Math libs"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.amd-llvm]
            artifact_group = "compiler"
            type = "target-neutral"

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            """
        )

    def test_nested_math_libs_path_maps_to_math_libs(self):
        self.write_nested_path_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["math-libs/libhipcxx/include/foo.hpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertIn("math-libs", result.matched_source_sets)
        self.assertIn("math-libs", result.impacted_artifact_groups)
        self.assertIn("math-libs", result.rebuild_stages)

    def test_nested_hipify_path_maps_to_compiler_runtime(self):
        self.write_nested_path_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["compiler/hipify/src/foo.cpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertIn("compilers", result.matched_source_sets)
        self.assertIn("compiler-runtime", result.rebuild_stages)

    def test_nested_compiler_path_maps_to_compiler_runtime(self):
        self.write_nested_path_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["compiler/amd-llvm/lib/Target/AMDGPU/foo.cpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertIn("compilers", result.matched_source_sets)
        self.assertIn("compiler", result.impacted_artifact_groups)
        self.assertIn("compiler-runtime", result.rebuild_stages)

    def test_stage_impact_result_shape(self):
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(["rocm-libraries"], topology=topology)

        payload = result.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {
                "changed_inputs",
                "matched_source_sets",
                "impacted_artifact_groups",
                "rebuild_stages",
                "copy_stages",
                "full_rebuild_required",
                "reasons",
                "unmatched_inputs",
                "impacted_artifacts",
                "reusable_artifacts",
                "artifact_level_analysis",
            },
        )
        self.assertEqual(payload["changed_inputs"], ("rocm-libraries",))
        self.assertEqual(payload["reasons"], ())
        self.assertEqual(payload["unmatched_inputs"], ())
        self.assertEqual(payload["matched_source_sets"], ("rocm-libraries",))
        self.assertFalse(payload["full_rebuild_required"])


class ArtifactLevelAnalysisTest(unittest.TestCase):
    """Test cases for granular artifact-level analysis."""

    def setUp(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as temp_file:
            self.topology_path = temp_file.name

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.topology_path):
            os.unlink(self.topology_path)

    def write_topology(self, content: str) -> None:
        """Write topology content to temp file."""
        with open(self.topology_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(content))

    def write_math_libs_topology(self) -> None:
        """Write a topology with multiple math-libs artifacts."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.blas]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocblas", "hipblas"]

            [artifacts.fft]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocfft", "hipfft"]

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocprim", "hipcub", "rocthrust"]

            [artifacts.rand]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocrand", "hiprand"]
            """
        )

    def test_rocblas_change_identifies_blas_artifact(self):
        """A rocblas change should identify blas as impacted artifact."""
        self.write_math_libs_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries/projects/rocblas/src/foo.cpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        self.assertIn("blas", result.impacted_artifacts)
        # Other artifacts should be reusable
        self.assertIn("fft", result.reusable_artifacts)
        self.assertIn("prim", result.reusable_artifacts)
        self.assertIn("rand", result.reusable_artifacts)

    def test_multiple_project_changes(self):
        """Multiple project changes should identify multiple impacted artifacts."""
        self.write_math_libs_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            [
                "rocm-libraries/projects/rocblas/src/foo.cpp",
                "rocm-libraries/projects/rocfft/src/bar.cpp",
            ],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        self.assertIn("blas", result.impacted_artifacts)
        self.assertIn("fft", result.impacted_artifacts)
        # prim and rand should still be reusable
        self.assertIn("prim", result.reusable_artifacts)
        self.assertIn("rand", result.reusable_artifacts)

    def test_shared_change_disables_artifact_analysis(self):
        """A shared/ change should disable artifact-level analysis."""
        self.write_math_libs_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries/shared/common.hpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        # Artifact-level analysis should NOT be enabled due to conservative fallback
        self.assertFalse(result.artifact_level_analysis)
        self.assertEqual(result.impacted_artifacts, ())
        self.assertEqual(result.reusable_artifacts, ())

    def test_submodule_root_disables_artifact_analysis(self):
        """A submodule root (no subpath) should not enable artifact analysis."""
        self.write_math_libs_topology()

        topology = BuildTopology(self.topology_path)
        # Just the submodule name, no specific file path
        result = analyze_stage_impact(
            ["rocm-libraries"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        # Cannot do artifact-level analysis without specific file paths
        self.assertFalse(result.artifact_level_analysis)

    def test_non_rocm_libraries_no_artifact_analysis(self):
        """Non-rocm-libraries changes should not enable artifact analysis."""
        self.write_topology(
            """
            [source_sets.compilers]
            description = "Compilers"
            submodules = ["llvm-project"]

            [artifact_groups.compiler]
            description = "Compiler"
            type = "generic"
            source_sets = ["compilers"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["compiler"]

            [artifacts.amd-llvm]
            artifact_group = "compiler"
            type = "target-neutral"
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["llvm-project/llvm/lib/Target/AMDGPU/foo.cpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        # No artifact-level analysis for non-rocm-libraries
        self.assertFalse(result.artifact_level_analysis)


class ImpactedArtifactStageNarrowingTest(unittest.TestCase):
    """Artifact-level hits narrow rebuild_stages so other stages can be copied."""

    def test_hrr_path_leaves_unrelated_stage_copyable(self):
        topology_text = textwrap.dedent(
            """
            [source_sets.rocm-systems]
            description = "ROCm systems"
            submodules = ["rocm-systems"]

            [artifact_groups.hip-runtime]
            description = "HIP runtime"
            type = "generic"
            source_sets = ["rocm-systems"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-systems"]

            [build_stages.compiler-runtime]
            description = "Compiler runtime"
            artifact_groups = ["hip-runtime"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]

            [artifacts.core-hip]
            artifact_group = "hip-runtime"
            type = "target-neutral"
            source_paths = ["clr", "hip"]

            [artifacts.core-hrr]
            artifact_group = "hip-runtime"
            type = "target-neutral"
            artifact_deps = ["core-hip"]
            source_paths = ["hrr"]

            [artifacts.blas]
            artifact_group = "math-libs"
            type = "target-specific"
            artifact_deps = ["core-hip"]
            source_paths = ["rocblas"]
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write(topology_text)
            path = handle.name
        try:
            topology = BuildTopology(path)
            result = analyze_stage_impact(
                ["rocm-systems/projects/hrr/playback/foo.cpp"],
                topology=topology,
            )
        finally:
            os.unlink(path)

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        self.assertEqual(result.impacted_artifacts, ("core-hrr",))
        self.assertEqual(result.rebuild_stages, ("compiler-runtime",))
        self.assertEqual(result.copy_stages, ("math-libs",))

    def test_real_topology_hrr_path_copies_math_libs(self):
        result = analyze_stage_impact(
            ["rocm-systems/projects/hrr/playback/foo.cpp"],
            topology=get_topology(),
        )
        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        self.assertIn("core-hrr", result.impacted_artifacts)
        self.assertIn("compiler-runtime", result.rebuild_stages)
        self.assertIn("math-libs", result.copy_stages)
        self.assertTrue(result.copy_stages)

    def _analyze_synthetic(self, topology_text, changed):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write(textwrap.dedent(topology_text))
            path = handle.name
        try:
            return analyze_stage_impact(changed, topology=BuildTopology(path))
        finally:
            os.unlink(path)

    def test_same_stage_consumer_still_pulls_in_its_own_consumers(self):
        # base and mid share stage-a; top (stage-b) depends on mid only. A base
        # change must rebuild stage-b through mid even though mid's stage is
        # already marked when the walk reaches it.
        result = self._analyze_synthetic(
            """
            [source_sets.mono]
            description = "Mono"
            submodules = ["mono"]

            [artifact_groups.ga]
            description = "A"
            type = "generic"
            source_sets = ["mono"]

            [artifact_groups.gb]
            description = "B"
            type = "generic"
            source_sets = ["mono"]

            [build_stages.stage-a]
            description = "A"
            artifact_groups = ["ga"]

            [build_stages.stage-b]
            description = "B"
            artifact_groups = ["gb"]

            [artifacts.base]
            artifact_group = "ga"
            type = "target-neutral"
            source_paths = ["base"]

            [artifacts.mid]
            artifact_group = "ga"
            type = "target-neutral"
            artifact_deps = ["base"]
            source_paths = ["mid"]

            [artifacts.top]
            artifact_group = "gb"
            type = "target-neutral"
            artifact_deps = ["mid"]
            source_paths = ["top"]
            """,
            ["mono/projects/base/foo.cpp"],
        )
        self.assertEqual(result.rebuild_stages, ("stage-a", "stage-b"))
        self.assertEqual(result.copy_stages, ())

    def test_real_topology_rocr_change_rebuilds_transitive_consumers(self):
        # core-runtime <- core-hip <- rdc (dctools-core) and rocdxg (wsl-rocdxg).
        result = analyze_stage_impact(
            ["rocm-systems/projects/rocr-runtime/foo.cpp"],
            topology=get_topology(),
            platform="linux",
        )
        self.assertEqual(
            result.rebuild_stages,
            (
                "comm-libs",
                "compiler-runtime",
                "cv-libs",
                "dctools-core",
                "debug-tools",
                "math-libs",
                "media-libs",
                "profiler-apps",
                "runtime-tests",
                "storage-libs",
                "wsl-rocdxg",
            ),
        )
        self.assertEqual(result.copy_stages, ("emulation",))

    def test_real_topology_clr_change_matches_rocr_change(self):
        clr = analyze_stage_impact(
            ["rocm-systems/projects/clr/foo.cpp"],
            topology=get_topology(),
            platform="linux",
        )
        self.assertNotIn("dctools-core", clr.copy_stages)
        self.assertNotIn("wsl-rocdxg", clr.copy_stages)
        self.assertIn("dctools-core", clr.rebuild_stages)
        self.assertIn("wsl-rocdxg", clr.rebuild_stages)
        self.assertEqual(clr.copy_stages, ("emulation",))

    def test_real_topology_hrr_change_rebuilds_only_compiler_runtime(self):
        result = analyze_stage_impact(
            ["rocm-systems/projects/hrr/playback/foo.cpp"],
            topology=get_topology(),
            platform="linux",
        )
        self.assertEqual(result.rebuild_stages, ("compiler-runtime",))
        self.assertIn("math-libs", result.copy_stages)
        self.assertIn("dctools-core", result.copy_stages)

    def test_real_topology_every_transitive_consumer_stage_is_rebuilt(self):
        # For each artifact, the rebuilt stages must contain the stage of every
        # transitive consumer. Computed here by fixpoint iteration over the
        # artifact_deps, independent of the implementation's worklist.
        topology = get_topology()
        from github_actions.stage_impact import StageImpactAnalyzer

        analyzer = StageImpactAnalyzer(topology)
        for name in topology.artifacts:
            closure = {name}
            changed = True
            while changed:
                changed = False
                for other in topology.artifacts.values():
                    if other.name not in closure and closure & set(
                        other.artifact_deps
                    ):
                        closure.add(other.name)
                        changed = True
            expected = {
                topology.get_stage_for_artifact(a)
                for a in closure
                if topology.get_stage_for_artifact(a)
            }
            with self.subTest(artifact=name):
                self.assertEqual(
                    analyzer._stages_for_impacted_artifacts({name}), expected
                )


class SharedSourcePathsAnalysisTest(unittest.TestCase):
    """Test cases for shared/ paths in source_paths artifact-level analysis."""

    def setUp(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as temp_file:
            self.topology_path = temp_file.name

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.topology_path):
            os.unlink(self.topology_path)

    def write_topology(self, content: str) -> None:
        """Write topology content to temp file."""
        with open(self.topology_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(content))

    def write_shared_source_paths_topology(self) -> None:
        """Write a topology with shared paths in source_paths."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.blas]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocblas", "hipblas", "origami", "blas-common"]

            [artifacts.fft]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocfft", "hipfft"]

            [artifacts.prim]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocprim", "hipcub", "rocthrust"]

            [artifacts.rand]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocrand", "hiprand"]
            """
        )

    def test_shared_path_maps_to_artifact(self):
        """A shared/ change with source_paths mapping should identify the artifact."""
        self.write_shared_source_paths_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries/shared/origami/foo.hpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        self.assertIn("blas", result.impacted_artifacts)
        # Other artifacts should be reusable
        self.assertIn("fft", result.reusable_artifacts)
        self.assertIn("prim", result.reusable_artifacts)
        self.assertIn("rand", result.reusable_artifacts)

    def test_unmapped_shared_disables_artifact_analysis(self):
        """A shared/ change without source_paths mapping should be conservative."""
        self.write_shared_source_paths_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries/shared/unknown-shared/foo.hpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        # Artifact-level analysis should NOT be enabled due to conservative fallback
        self.assertFalse(result.artifact_level_analysis)
        self.assertEqual(result.impacted_artifacts, ())
        self.assertEqual(result.reusable_artifacts, ())

    def test_shared_source_path_with_multiple_artifacts(self):
        """A source_path used by multiple artifacts should impact all of them."""
        self.write_topology(
            """
            [source_sets.rocm-libraries]
            description = "ROCm libraries"
            submodules = ["rocm-libraries"]

            [artifact_groups.math-libs]
            description = "Math libs"
            type = "per-arch"
            source_sets = ["rocm-libraries"]

            [build_stages.math-libs]
            description = "Math libs stage"
            artifact_groups = ["math-libs"]
            type = "per-arch"

            [artifacts.blas]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocblas", "common-utils"]

            [artifacts.fft]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocfft", "common-utils"]

            [artifacts.rand]
            artifact_group = "math-libs"
            type = "target-specific"
            source_paths = ["rocrand"]
            """
        )

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            ["rocm-libraries/shared/common-utils/helper.hpp"],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        # Both blas and fft should be impacted
        self.assertIn("blas", result.impacted_artifacts)
        self.assertIn("fft", result.impacted_artifacts)
        # Only rand should be reusable
        self.assertIn("rand", result.reusable_artifacts)
        self.assertEqual(len(result.reusable_artifacts), 1)

    def test_mixed_project_and_shared_changes(self):
        """Mixed project and shared changes should union impacted artifacts."""
        self.write_shared_source_paths_topology()

        topology = BuildTopology(self.topology_path)
        result = analyze_stage_impact(
            [
                "rocm-libraries/projects/rocfft/src/foo.cpp",
                "rocm-libraries/shared/origami/bar.hpp",
            ],
            topology=topology,
        )

        self.assertFalse(result.full_rebuild_required)
        self.assertTrue(result.artifact_level_analysis)
        # fft from project path, blas from source_paths
        self.assertIn("fft", result.impacted_artifacts)
        self.assertIn("blas", result.impacted_artifacts)
        # prim and rand should be reusable
        self.assertIn("prim", result.reusable_artifacts)
        self.assertIn("rand", result.reusable_artifacts)


if __name__ == "__main__":
    unittest.main()
