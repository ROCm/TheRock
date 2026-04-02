#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for build_topology module.
"""

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.build_topology import (
    BuildStage,
    ArtifactGroup,
    Artifact,
    BuildTopology,
)


class BuildTopologyTest(unittest.TestCase):
    """Test cases for BuildTopology class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file and close it immediately to avoid file locking on Windows
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as temp_file:
            self.topology_path = temp_file.name

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.topology_path):
            os.unlink(self.topology_path)

    def write_topology(self, content: str):
        """Write topology content to temp file."""
        with open(self.topology_path, "w") as f:
            f.write(textwrap.dedent(content))

    def test_empty_topology(self):
        """Test parsing an empty topology file."""
        self.write_topology(
            """
            [metadata]
            version = "1.0"
        """
        )

        topology = BuildTopology(self.topology_path)
        self.assertEqual(len(topology.get_build_stages()), 0)
        self.assertEqual(len(topology.get_artifact_groups()), 0)
        self.assertEqual(len(topology.get_artifacts()), 0)

    def test_parse_build_stages(self):
        """Test parsing build stages."""
        self.write_topology(
            """
            [build_stages.foundation]
            description = "Foundation stage"
            artifact_groups = ["base", "sysdeps"]

            [build_stages.compiler]
            description = "Compiler stage"
            artifact_groups = ["llvm"]
            type = "per-arch"
        """
        )

        topology = BuildTopology(self.topology_path)
        stages = topology.get_build_stages()

        self.assertEqual(len(stages), 2)

        foundation = topology.build_stages["foundation"]
        self.assertEqual(foundation.name, "foundation")
        self.assertEqual(foundation.description, "Foundation stage")
        self.assertEqual(foundation.artifact_groups, ["base", "sysdeps"])
        self.assertEqual(foundation.type, "generic")

        compiler = topology.build_stages["compiler"]
        self.assertEqual(compiler.type, "per-arch")

    def test_parse_artifact_groups(self):
        """Test parsing artifact groups."""
        self.write_topology(
            """
            [artifact_groups.base]
            description = "Base infrastructure"
            type = "generic"

            [artifact_groups.runtime]
            description = "Runtime components"
            type = "generic"
            artifact_group_deps = ["base"]
        """
        )

        topology = BuildTopology(self.topology_path)
        groups = topology.get_artifact_groups()

        self.assertEqual(len(groups), 2)

        base = topology.artifact_groups["base"]
        self.assertEqual(base.name, "base")
        self.assertEqual(base.description, "Base infrastructure")
        self.assertEqual(base.type, "generic")
        self.assertEqual(base.artifact_group_deps, [])

        runtime = topology.artifact_groups["runtime"]
        self.assertEqual(runtime.artifact_group_deps, ["base"])

    def test_parse_artifacts(self):
        """Test parsing artifacts."""
        self.write_topology(
            """
            [artifacts.rocm-core]
            artifact_group = "base"
            type = "target-neutral"

            [artifacts.hip]
            artifact_group = "runtime"
            type = "target-specific"
            artifact_deps = ["rocm-core"]
            platform = "linux"
        """
        )

        topology = BuildTopology(self.topology_path)
        artifacts = topology.get_artifacts()

        self.assertEqual(len(artifacts), 2)

        rocm_core = topology.artifacts["rocm-core"]
        self.assertEqual(rocm_core.name, "rocm-core")
        self.assertEqual(rocm_core.artifact_group, "base")
        self.assertEqual(rocm_core.type, "target-neutral")
        self.assertEqual(rocm_core.artifact_deps, [])
        self.assertIsNone(rocm_core.platform)

        hip = topology.artifacts["hip"]
        self.assertEqual(hip.artifact_group, "runtime")
        self.assertEqual(hip.type, "target-specific")
        self.assertEqual(hip.artifact_deps, ["rocm-core"])
        self.assertEqual(hip.platform, "linux")

    def test_get_artifacts_in_group(self):
        """Test getting artifacts belonging to a group."""
        self.write_topology(
            """
            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.artifact2]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.artifact3]
            artifact_group = "group2"
            type = "target-neutral"
        """
        )

        topology = BuildTopology(self.topology_path)

        group1_artifacts = topology.get_artifacts_in_group("group1")
        self.assertEqual(len(group1_artifacts), 2)
        self.assertIn("artifact1", [a.name for a in group1_artifacts])
        self.assertIn("artifact2", [a.name for a in group1_artifacts])

        group2_artifacts = topology.get_artifacts_in_group("group2")
        self.assertEqual(len(group2_artifacts), 1)
        self.assertEqual(group2_artifacts[0].name, "artifact3")

    def test_get_produced_artifacts(self):
        """Test getting artifacts produced by a build stage."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1", "group2"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifact_groups.group2]
            description = "Group 2"
            type = "generic"

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.artifact2]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.artifact3]
            artifact_group = "group2"
            type = "target-neutral"

            [artifacts.artifact4]
            artifact_group = "group3"
            type = "target-neutral"
        """
        )

        topology = BuildTopology(self.topology_path)
        produced = topology.get_produced_artifacts("stage1")

        self.assertEqual(len(produced), 3)
        self.assertIn("artifact1", produced)
        self.assertIn("artifact2", produced)
        self.assertIn("artifact3", produced)
        self.assertNotIn("artifact4", produced)

    def test_get_inbound_artifacts(self):
        """Test getting inbound artifacts for a build stage."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [build_stages.stage2]
            description = "Stage 2"
            artifact_groups = ["group2"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifact_groups.group2]
            description = "Group 2"
            type = "generic"
            artifact_group_deps = ["group1"]

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.artifact2]
            artifact_group = "group2"
            type = "target-neutral"
            artifact_deps = ["artifact1"]
        """
        )

        topology = BuildTopology(self.topology_path)

        # Stage1 has no inbound artifacts
        stage1_inbound = topology.get_inbound_artifacts("stage1")
        self.assertEqual(len(stage1_inbound), 0)

        # Stage2 depends on artifacts from stage1
        stage2_inbound = topology.get_inbound_artifacts("stage2")
        self.assertEqual(len(stage2_inbound), 1)
        self.assertIn("artifact1", stage2_inbound)

    def test_validate_missing_references(self):
        """Test validation catches missing references."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage with missing group"
            artifact_groups = ["missing_group"]

            [artifact_groups.group1]
            description = "Group with missing dependency"
            type = "generic"
            artifact_group_deps = ["missing_dep"]

            [artifacts.artifact1]
            artifact_group = "missing_artifact_group"
            type = "target-neutral"
            artifact_deps = ["missing_artifact"]
        """
        )

        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()

        self.assertGreater(len(errors), 0)
        self.assertTrue(any("missing_group" in e for e in errors))
        self.assertTrue(any("missing_dep" in e for e in errors))
        self.assertTrue(any("missing_artifact_group" in e for e in errors))
        self.assertTrue(any("missing_artifact" in e for e in errors))

    def test_validate_circular_dependencies(self):
        """Test validation catches circular dependencies."""
        self.write_topology(
            """
            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"
            artifact_group_deps = ["group2"]

            [artifact_groups.group2]
            description = "Group 2"
            type = "generic"
            artifact_group_deps = ["group3"]

            [artifact_groups.group3]
            description = "Group 3"
            type = "generic"
            artifact_group_deps = ["group1"]
        """
        )

        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()

        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Circular dependency" in e for e in errors))

    def test_get_build_order(self):
        """Test getting the build order based on dependencies."""
        self.write_topology(
            """
            [build_stages.foundation]
            description = "Foundation"
            artifact_groups = ["base"]

            [build_stages.compiler]
            description = "Compiler"
            artifact_groups = ["llvm"]

            [build_stages.runtime]
            description = "Runtime"
            artifact_groups = ["hip"]

            [artifact_groups.base]
            description = "Base"
            type = "generic"

            [artifact_groups.llvm]
            description = "LLVM"
            type = "generic"
            artifact_group_deps = ["base"]

            [artifact_groups.hip]
            description = "HIP"
            type = "generic"
            artifact_group_deps = ["llvm"]
        """
        )

        topology = BuildTopology(self.topology_path)
        build_order = topology.get_build_order()

        # Foundation should come before compiler
        foundation_idx = build_order.index("foundation")
        compiler_idx = build_order.index("compiler")
        self.assertLess(foundation_idx, compiler_idx)

        # Compiler should come before runtime
        runtime_idx = build_order.index("runtime")
        self.assertLess(compiler_idx, runtime_idx)

    def test_get_dependency_graph(self):
        """Test generating dependency graph."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"
        """
        )

        topology = BuildTopology(self.topology_path)
        graph = topology.get_dependency_graph()

        self.assertIn("build_stages", graph)
        self.assertIn("artifact_groups", graph)
        self.assertIn("artifacts", graph)

        self.assertIn("stage1", graph["build_stages"])
        self.assertIn("group1", graph["artifact_groups"])
        self.assertIn("artifact1", graph["artifacts"])

        # Check stage details
        stage1_data = graph["build_stages"]["stage1"]
        self.assertEqual(stage1_data["type"], "generic")
        self.assertEqual(stage1_data["artifact_groups"], ["group1"])
        self.assertIn("produced_artifacts", stage1_data)
        self.assertIn("artifact1", stage1_data["produced_artifacts"])

    def test_invalid_stage_name(self):
        """Test handling of invalid stage name."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = []
        """
        )

        topology = BuildTopology(self.topology_path)

        with self.assertRaises(ValueError) as context:
            topology.get_inbound_artifacts("nonexistent_stage")

        self.assertIn("not found", str(context.exception))

    def test_diamond_dependency_pattern(self):
        """Test diamond dependency pattern doesn't cause redundant processing."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [build_stages.stage2]
            description = "Stage 2"
            artifact_groups = ["group2"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifact_groups.group2]
            description = "Group 2"
            type = "generic"
            artifact_group_deps = ["group1"]

            # Diamond pattern:
            #     A
            #    / \\
            #   B   C
            #    \\ /
            #     D
            [artifacts.D]
            artifact_group = "group1"
            type = "target-neutral"

            [artifacts.B]
            artifact_group = "group1"
            type = "target-neutral"
            artifact_deps = ["D"]

            [artifacts.C]
            artifact_group = "group1"
            type = "target-neutral"
            artifact_deps = ["D"]

            [artifacts.A]
            artifact_group = "group2"
            type = "target-neutral"
            artifact_deps = ["B", "C"]
        """
        )

        topology = BuildTopology(self.topology_path)

        # Stage2 should get D only once, not twice
        stage2_inbound = topology.get_inbound_artifacts("stage2")

        # Count how many times D appears (should be exactly once)
        d_count = list(stage2_inbound).count("D")
        self.assertEqual(d_count, 1, "D should appear exactly once in dependencies")

        # Should have all three dependencies B, C, D
        self.assertEqual(len(stage2_inbound), 3)
        self.assertIn("B", stage2_inbound)
        self.assertIn("C", stage2_inbound)
        self.assertIn("D", stage2_inbound)

    def test_complex_dependency_chain(self):
        """Test complex dependency chain resolution."""
        self.write_topology(
            """
            [build_stages.foundation]
            artifact_groups = ["base"]
            description = "Foundation"

            [build_stages.compiler]
            artifact_groups = ["llvm"]
            description = "Compiler"

            [build_stages.runtime]
            artifact_groups = ["hip"]
            description = "Runtime"

            [build_stages.libraries]
            artifact_groups = ["math"]
            description = "Libraries"

            [artifact_groups.base]
            type = "generic"
            description = "Base"

            [artifact_groups.llvm]
            type = "generic"
            artifact_group_deps = ["base"]
            description = "LLVM"

            [artifact_groups.hip]
            type = "generic"
            artifact_group_deps = ["base", "llvm"]
            description = "HIP"

            [artifact_groups.math]
            type = "generic"
            artifact_group_deps = ["hip"]
            description = "Math"

            [artifacts.base-artifact]
            artifact_group = "base"
            type = "target-neutral"

            [artifacts.llvm-artifact]
            artifact_group = "llvm"
            type = "target-neutral"
            artifact_deps = ["base-artifact"]

            [artifacts.hip-artifact]
            artifact_group = "hip"
            type = "target-neutral"
            artifact_deps = ["base-artifact", "llvm-artifact"]

            [artifacts.math-artifact]
            artifact_group = "math"
            type = "target-neutral"
            artifact_deps = ["hip-artifact"]
        """
        )

        topology = BuildTopology(self.topology_path)

        # Libraries stage should need all upstream artifacts
        libs_inbound = topology.get_inbound_artifacts("libraries")
        self.assertIn("hip-artifact", libs_inbound)
        self.assertIn("llvm-artifact", libs_inbound)
        self.assertIn("base-artifact", libs_inbound)

        # Foundation stage should need nothing
        foundation_inbound = topology.get_inbound_artifacts("foundation")
        self.assertEqual(len(foundation_inbound), 0)

    def test_get_artifact_feature_name_default(self):
        """Test default feature name derivation from artifact name."""
        self.write_topology(
            """
            [artifacts.rocm-core]
            artifact_group = "base"
            type = "target-neutral"
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["rocm-core"]
        feature_name = topology.get_artifact_feature_name(artifact)
        self.assertEqual(feature_name, "ROCM_CORE")

    def test_get_artifact_feature_name_override(self):
        """Test explicit feature_name override."""
        self.write_topology(
            """
            [artifacts.rocm-core]
            artifact_group = "base"
            type = "target-neutral"
            feature_name = "MY_CUSTOM_NAME"
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["rocm-core"]
        feature_name = topology.get_artifact_feature_name(artifact)
        self.assertEqual(feature_name, "MY_CUSTOM_NAME")

    def test_get_artifact_feature_group_default(self):
        """Test default feature group derivation from artifact group."""
        self.write_topology(
            """
            [artifact_groups.math-libs]
            description = "Math libraries"
            type = "generic"

            [artifacts.rocblas]
            artifact_group = "math-libs"
            type = "target-neutral"
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["rocblas"]
        feature_group = topology.get_artifact_feature_group(artifact)
        self.assertEqual(feature_group, "MATH_LIBS")

    def test_get_artifact_feature_group_override(self):
        """Test explicit feature_group override."""
        self.write_topology(
            """
            [artifacts.rocblas]
            artifact_group = "math-libs"
            type = "target-neutral"
            feature_group = "CUSTOM_GROUP"
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["rocblas"]
        feature_group = topology.get_artifact_feature_group(artifact)
        self.assertEqual(feature_group, "CUSTOM_GROUP")

    def test_get_submodules_for_source_set(self):
        """Test getting submodules for a specific source set."""
        self.write_topology(
            """
            [source_sets.compiler-sources]
            description = "Compiler sources"
            submodules = ["llvm-project", "device-libs"]
        """
        )
        topology = BuildTopology(self.topology_path)
        submodules = topology.get_submodules_for_source_set("compiler-sources")
        self.assertEqual(len(submodules), 2)
        names = [s.name for s in submodules]
        self.assertIn("llvm-project", names)
        self.assertIn("device-libs", names)

    def test_get_submodules_for_source_set_not_found(self):
        """Test error when source set doesn't exist."""
        self.write_topology(
            """
            [metadata]
            version = "1.0"
        """
        )
        topology = BuildTopology(self.topology_path)
        with self.assertRaises(ValueError) as ctx:
            topology.get_submodules_for_source_set("nonexistent")
        self.assertIn("not found", str(ctx.exception))

    def test_get_submodules_for_stage(self):
        """Test getting submodules needed for a build stage."""
        self.write_topology(
            """
            [source_sets.base-sources]
            description = "Base sources"
            submodules = ["rocm-cmake", "rocm-core"]

            [source_sets.compiler-sources]
            description = "Compiler sources"
            submodules = ["llvm-project"]

            [build_stages.compiler]
            description = "Compiler stage"
            artifact_groups = ["compiler-group"]

            [artifact_groups.compiler-group]
            description = "Compiler group"
            type = "generic"
            source_sets = ["base-sources", "compiler-sources"]
        """
        )
        topology = BuildTopology(self.topology_path)
        submodules = topology.get_submodules_for_stage("compiler")
        names = [s.name for s in submodules]
        self.assertIn("rocm-cmake", names)
        self.assertIn("rocm-core", names)
        self.assertIn("llvm-project", names)
        self.assertEqual(len(names), 3)

    def test_get_submodules_for_stage_deduplication(self):
        """Test that overlapping source sets deduplicate submodules."""
        self.write_topology(
            """
            [source_sets.set-a]
            description = "Set A"
            submodules = ["shared-mod", "mod-a"]

            [source_sets.set-b]
            description = "Set B"
            submodules = ["shared-mod", "mod-b"]

            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"
            source_sets = ["set-a", "set-b"]
        """
        )
        topology = BuildTopology(self.topology_path)
        submodules = topology.get_submodules_for_stage("stage1")
        names = [s.name for s in submodules]
        # shared-mod should appear only once
        self.assertEqual(names.count("shared-mod"), 1)
        self.assertEqual(len(names), 3)

    def test_get_submodules_for_stage_platform_filter(self):
        """Test that platform-disabled source sets are skipped."""
        self.write_topology(
            """
            [source_sets.linux-only]
            description = "Linux only sources"
            submodules = ["linux-mod"]
            disable_platforms = ["windows"]

            [source_sets.common]
            description = "Common sources"
            submodules = ["common-mod"]

            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"
            source_sets = ["linux-only", "common"]
        """
        )
        topology = BuildTopology(self.topology_path)

        # On linux, both source sets should be included
        linux_mods = topology.get_submodules_for_stage("stage1", platform="linux")
        linux_names = [s.name for s in linux_mods]
        self.assertIn("linux-mod", linux_names)
        self.assertIn("common-mod", linux_names)

        # On windows, linux-only source set should be skipped
        win_mods = topology.get_submodules_for_stage("stage1", platform="windows")
        win_names = [s.name for s in win_mods]
        self.assertNotIn("linux-mod", win_names)
        self.assertIn("common-mod", win_names)

    def test_get_submodules_for_stage_not_found(self):
        """Test error when build stage doesn't exist."""
        self.write_topology(
            """
            [metadata]
            version = "1.0"
        """
        )
        topology = BuildTopology(self.topology_path)
        with self.assertRaises(ValueError):
            topology.get_submodules_for_stage("nonexistent")

    def test_get_python_requires_for_stage(self):
        """Test getting python_requires for artifacts in a stage."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"
            python_requires = ["-r requirements.txt", "mako"]

            [artifacts.artifact2]
            artifact_group = "group1"
            type = "target-neutral"
            python_requires = ["pyyaml"]
        """
        )
        topology = BuildTopology(self.topology_path)
        requires = topology.get_python_requires_for_stage("stage1")
        self.assertIn("-r requirements.txt", requires)
        self.assertIn("mako", requires)
        self.assertIn("pyyaml", requires)
        self.assertEqual(len(requires), 3)

    def test_get_python_requires_for_stage_deduplication(self):
        """Test that duplicate python_requires are deduplicated."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"
            python_requires = ["mako", "pyyaml"]

            [artifacts.artifact2]
            artifact_group = "group1"
            type = "target-neutral"
            python_requires = ["mako", "jinja2"]
        """
        )
        topology = BuildTopology(self.topology_path)
        requires = topology.get_python_requires_for_stage("stage1")
        # mako should appear only once
        self.assertEqual(requires.count("mako"), 1)
        self.assertEqual(len(requires), 3)

    def test_get_python_requires_for_stage_not_found(self):
        """Test error when stage doesn't exist."""
        self.write_topology(
            """
            [metadata]
            version = "1.0"
        """
        )
        topology = BuildTopology(self.topology_path)
        with self.assertRaises(ValueError):
            topology.get_python_requires_for_stage("nonexistent")

    def test_get_python_requires_for_stage_empty(self):
        """Test stage with no python_requires."""
        self.write_topology(
            """
            [build_stages.stage1]
            description = "Stage 1"
            artifact_groups = ["group1"]

            [artifact_groups.group1]
            description = "Group 1"
            type = "generic"

            [artifacts.artifact1]
            artifact_group = "group1"
            type = "target-neutral"
        """
        )
        topology = BuildTopology(self.topology_path)
        requires = topology.get_python_requires_for_stage("stage1")
        self.assertEqual(requires, [])

    def test_validate_naming_conventions_valid(self):
        """Test validation passes for correctly named entities."""
        self.write_topology(
            """
            [build_stages.my-stage]
            description = "A stage"
            artifact_groups = ["my-group"]

            [artifact_groups.my-group]
            description = "A group"
            type = "generic"

            [artifacts.my-artifact]
            artifact_group = "my-group"
            type = "target-neutral"
            feature_name = "MY_ARTIFACT"
            feature_group = "MY_GROUP"
        """
        )
        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()
        # Filter to only naming convention errors
        naming_errors = [
            e
            for e in errors
            if "should be" in e or "invalid type" in e or "invalid platform" in e
        ]
        self.assertEqual(naming_errors, [])

    def test_validate_naming_conventions_invalid_names(self):
        """Test validation catches invalid entity names."""
        self.write_topology(
            """
            [build_stages.MyStage]
            description = "Bad name"
            artifact_groups = []

            [artifact_groups.My_Group]
            description = "Bad name"
            type = "generic"

            [artifacts.BadArtifact]
            artifact_group = "My_Group"
            type = "target-neutral"
        """
        )
        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()
        self.assertTrue(any("MyStage" in e and "lowercase" in e for e in errors))
        self.assertTrue(any("My_Group" in e and "lowercase" in e for e in errors))
        self.assertTrue(any("BadArtifact" in e and "lowercase" in e for e in errors))

    def test_validate_naming_conventions_invalid_feature_name(self):
        """Test validation catches invalid feature_name format."""
        self.write_topology(
            """
            [artifacts.my-artifact]
            artifact_group = "group"
            type = "target-neutral"
            feature_name = "bad-name"
        """
        )
        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()
        self.assertTrue(
            any("feature_name" in e and "UPPERCASE" in e for e in errors)
        )

    def test_validate_naming_conventions_invalid_platform(self):
        """Test validation catches invalid platform values."""
        self.write_topology(
            """
            [artifacts.my-artifact]
            artifact_group = "group"
            type = "target-neutral"
            platform = "macos"
        """
        )
        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()
        self.assertTrue(any("invalid platform" in e for e in errors))

    def test_get_all_submodules(self):
        """Test getting all submodules across all source sets."""
        self.write_topology(
            """
            [source_sets.set-a]
            description = "Set A"
            submodules = ["mod-a", "shared"]

            [source_sets.set-b]
            description = "Set B"
            submodules = ["mod-b", "shared"]
        """
        )
        topology = BuildTopology(self.topology_path)
        all_mods = topology.get_all_submodules()
        names = [s.name for s in all_mods]
        self.assertIn("mod-a", names)
        self.assertIn("mod-b", names)
        self.assertIn("shared", names)
        # shared should appear only once
        self.assertEqual(names.count("shared"), 1)

    def test_get_source_sets(self):
        """Test getting all source sets."""
        self.write_topology(
            """
            [source_sets.set-a]
            description = "Set A"
            submodules = ["mod-a"]

            [source_sets.set-b]
            description = "Set B"
            submodules = ["mod-b"]
        """
        )
        topology = BuildTopology(self.topology_path)
        source_sets = topology.get_source_sets()
        self.assertEqual(len(source_sets), 2)
        names = [ss.name for ss in source_sets]
        self.assertIn("set-a", names)
        self.assertIn("set-b", names)

    def test_artifact_disable_platforms(self):
        """Test parsing of artifact disable_platforms."""
        self.write_topology(
            """
            [artifacts.my-artifact]
            artifact_group = "group"
            type = "target-neutral"
            disable_platforms = ["windows"]
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["my-artifact"]
        self.assertEqual(artifact.disable_platforms, ["windows"])

    def test_artifact_split_databases(self):
        """Test parsing of artifact split_databases."""
        self.write_topology(
            """
            [artifacts.my-artifact]
            artifact_group = "group"
            type = "target-neutral"
            split_databases = ["rocblas", "hipblaslt"]
        """
        )
        topology = BuildTopology(self.topology_path)
        artifact = topology.artifacts["my-artifact"]
        self.assertEqual(artifact.split_databases, ["rocblas", "hipblaslt"])

    def test_python_requires_must_be_list(self):
        """Test that python_requires as non-list raises ValueError."""
        self.write_topology(
            """
            [artifacts.my-artifact]
            artifact_group = "group"
            type = "target-neutral"
            python_requires = "not-a-list"
        """
        )
        with self.assertRaises(ValueError) as ctx:
            BuildTopology(self.topology_path)
        self.assertIn("python_requires must be a list", str(ctx.exception))

    def test_circular_artifact_dependency(self):
        """Test validation catches circular artifact dependencies."""
        self.write_topology(
            """
            [artifacts.artifact-a]
            artifact_group = "group"
            type = "target-neutral"
            artifact_deps = ["artifact-b"]

            [artifacts.artifact-b]
            artifact_group = "group"
            type = "target-neutral"
            artifact_deps = ["artifact-a"]
        """
        )
        topology = BuildTopology(self.topology_path)
        errors = topology.validate_topology()
        self.assertTrue(any("Circular dependency" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
