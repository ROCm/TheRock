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

from _therock_utils.build_topology import BuildTopology
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
            },
        )
        self.assertEqual(payload["changed_inputs"], ("rocm-libraries",))
        self.assertEqual(payload["reasons"], ())
        self.assertEqual(payload["unmatched_inputs"], ())
        self.assertEqual(payload["matched_source_sets"], ("rocm-libraries",))
        self.assertFalse(payload["full_rebuild_required"])


if __name__ == "__main__":
    unittest.main()
