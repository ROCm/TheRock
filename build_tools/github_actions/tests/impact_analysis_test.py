# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import impact_analysis as ti


class _FakeStage:
    def __init__(self, artifact_groups):
        self.artifact_groups = artifact_groups


class _FakeSourceSet:
    def __init__(self, name):
        self.name = name


class _FakeStageImpact:
    def __init__(
        self,
        *,
        rebuild_stages=(),
        copy_stages=(),
        full_rebuild_required=False,
        reasons=(),
    ):
        self.rebuild_stages = tuple(rebuild_stages)
        self.copy_stages = tuple(copy_stages)
        self.full_rebuild_required = full_rebuild_required
        self.reasons = tuple(reasons)


class FakeTopology:
    def __init__(self):
        self.build_stages = {
            "compiler-runtime": _FakeStage(["base-group"]),
            "math-libs": _FakeStage(["blas-group"]),
            "media-libs": _FakeStage(["media-group"]),
            "debug-tools": _FakeStage(["debug-group"]),
            "comm-libs": _FakeStage(["comm-group"]),
        }
        self._artifact_groups = {
            "base-group": ["base"],
            "blas-group": ["blas"],
            "media-group": ["media"],
            "debug-group": ["dbg"],
            "comm-group": ["comm"],
        }

    def get_artifact_group_to_artifacts(self):
        return self._artifact_groups

    def get_produced_artifacts(self, stage_name):
        produced = {
            "compiler-runtime": {"base"},
            "math-libs": {"blas"},
            "media-libs": {"media"},
            "debug-tools": {"dbg"},
            "comm-libs": {"comm"},
        }
        return produced.get(stage_name, set())

    def get_source_set_for_path(self, path, platform=None):
        mapping = {
            "rocm-libraries/projects/rocBLAS/x.cpp": _FakeSourceSet("math"),
            "build_tools/foo.py": _FakeSourceSet("core"),
            "rocm-libraries/projects/rocJPEG/y.cpp": _FakeSourceSet("media"),
            "rocm-libraries/projects/rocgdb/z.cpp": _FakeSourceSet("debug"),
            "rocm-libraries/projects/rccl/a.cpp": _FakeSourceSet("comm"),
        }
        return mapping.get(path)

    def get_source_set_for_submodule(self, name, platform=None):
        mapping = {
            "rocm-libraries": _FakeSourceSet("math"),
            "build_tools": _FakeSourceSet("core"),
        }
        return mapping.get(name)


class TestMapStagesToTestComponents(unittest.TestCase):
    def test_compiler_runtime_maps_to_core_tests(self):
        comps = ti.map_stages_to_test_components(FakeTopology(), ["compiler-runtime"])
        self.assertIn("hip-tests", comps)
        self.assertIn("rocrtst", comps)
        self.assertIn("rocgdb", comps)
        self.assertIn("rocr-debug-agent", comps)

    def test_media_and_comm_stages_map(self):
        comps = ti.map_stages_to_test_components(
            FakeTopology(),
            ["media-libs", "comm-libs"],
        )
        self.assertIn("rocdecode", comps)
        self.assertIn("rocjpeg", comps)
        self.assertIn("rccl", comps)


class TestMapArtifactsToTestComponents(unittest.TestCase):
    def test_artifact_names_map_to_related_tests(self):
        comps = ti.map_artifacts_to_test_components(
            FakeTopology(),
            ["base", "amd-dbgapi", "rocdecode", "rccl"],
        )
        self.assertIn("hip-tests", comps)
        self.assertIn("rocgdb", comps)
        self.assertIn("rocdecode", comps)
        self.assertIn("rccl", comps)


class TestComputeTestMatrixFilter(unittest.TestCase):
    def test_dry_run_keeps_full_matrix_but_reports_affects(self):
        impact = _FakeStageImpact(
            rebuild_stages=("compiler-runtime",),
            copy_stages=("media-libs",),
            full_rebuild_required=False,
            reasons=(),
        )
        plan = ti.compute_test_matrix_filter(
            changed_paths=["rocm-libraries/projects/rocBLAS/x.cpp"],
            stage_impact_result=impact,
            topology=FakeTopology(),
            dry_run=True,
        )

        self.assertFalse(plan.full_rebuild_required)
        self.assertIn("hip-tests", plan.affected_test_components)
        self.assertIn("rocprofiler-sdk", plan.affected_test_components)
        self.assertEqual(plan.selected_test_components, ti.ALL_TEST_COMPONENTS)
        self.assertIn("dry-run only", "\n".join(plan.report_lines))

    def test_full_rebuild_falls_back_to_full_matrix(self):
        impact = _FakeStageImpact(
            rebuild_stages=(),
            copy_stages=(),
            full_rebuild_required=True,
            reasons=("build tooling changed",),
        )
        plan = ti.compute_test_matrix_filter(
            changed_paths=["build_tools/foo.py"],
            stage_impact_result=impact,
            topology=FakeTopology(),
            dry_run=True,
        )

        self.assertTrue(plan.full_rebuild_required)
        self.assertEqual(plan.selected_test_components, ti.ALL_TEST_COMPONENTS)
        self.assertIn("full CI fallback", "\n".join(plan.report_lines))

    def test_opt_in_mode_selects_only_affected_components(self):
        impact = _FakeStageImpact(
            rebuild_stages=("media-libs",),
            copy_stages=(),
            full_rebuild_required=False,
            reasons=(),
        )
        plan = ti.compute_test_matrix_filter(
            changed_paths=["rocm-libraries/projects/rocJPEG/y.cpp"],
            stage_impact_result=impact,
            topology=FakeTopology(),
            dry_run=False,
        )

        self.assertIn("rocdecode", plan.selected_test_components)
        self.assertIn("rocjpeg", plan.selected_test_components)
        self.assertNotIn("rccl", plan.selected_test_components)
        self.assertIn("rccl", plan.skipped_test_components)


if __name__ == "__main__":
    unittest.main()
