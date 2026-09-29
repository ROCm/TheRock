# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import configure_coverage_ci
from _therock_utils.build_topology import get_topology


def _fake_rocm_systems_project() -> configure_coverage_ci.CoverageProject:
    return configure_coverage_ci.CoverageProject(
        cmake_target="fakeSystemsLib",
        coverage_option="BUILD_CODE_COVERAGE",
        stage=configure_coverage_ci.STAGE_MATH_LIBS,
        test_component="fakesystemslib",
        coverage_config="projects/fakesystemslib/test_categories_coverage.yaml",
        source_repo=configure_coverage_ci.ROCM_SYSTEMS,
    )


class ParseProjectsTest(unittest.TestCase):
    def test_empty_selects_every_registered_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects(""),
            sorted(configure_coverage_ci.COVERAGE_PROJECTS),
        )

    def test_whitespace_and_case_are_normalized(self):
        self.assertEqual(configure_coverage_ci.parse_projects(" HipRand "), ["hiprand"])

    def test_duplicates_are_dropped_but_order_is_kept(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("hiprand,hiprand"), ["hiprand"]
        )

    def test_unknown_project_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            configure_coverage_ci.parse_projects("hiprand,not-a-project")
        self.assertIn("not-a-project", str(context.exception))

    def test_all_alias_expands_to_every_registered_project(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("all"),
            sorted(configure_coverage_ci.COVERAGE_PROJECTS),
        )

    def test_rocm_libraries_all_expands_to_that_group(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
            sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS),
        )

    def test_group_aliases_are_case_insensitive(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("  ALL  "),
            configure_coverage_ci.parse_projects("all"),
        )
        self.assertEqual(
            configure_coverage_ci.parse_projects("Rocm_Libraries_All"),
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
        )

    def test_alias_and_explicit_name_overlap_is_deduped(self):
        self.assertEqual(
            configure_coverage_ci.parse_projects("rocm_libraries_all,hiprand"),
            configure_coverage_ci.parse_projects("rocm_libraries_all"),
        )

    def test_empty_group_alias_is_rejected(self):
        # No rocm-systems project is registered yet. Selecting nothing is a
        # caller mistake worth reporting rather than silently producing no jobs.
        self.assertFalse(configure_coverage_ci.ROCM_SYSTEMS_PROJECTS)
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_projects("rocm_systems_all")

    def test_unknown_alias_like_token_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_projects("rocm_everything_all")


class SourceRepoPartitionTest(unittest.TestCase):
    def test_groups_partition_the_registered_projects(self):
        libraries = set(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
        systems = set(configure_coverage_ci.ROCM_SYSTEMS_PROJECTS)
        self.assertTrue(libraries.isdisjoint(systems))
        self.assertEqual(
            libraries | systems, set(configure_coverage_ci.COVERAGE_PROJECTS)
        )


class ParseAmdgpuFamiliesTest(unittest.TestCase):
    def test_empty_falls_back_to_the_default_family(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families(""),
            [configure_coverage_ci.DEFAULT_AMDGPU_FAMILIES],
        )

    def test_comma_separated_families_are_split(self):
        self.assertEqual(
            configure_coverage_ci.parse_amdgpu_families("gfx94X-dcgpu, gfx110X-all"),
            ["gfx94X-dcgpu", "gfx110X-all"],
        )


class ParseConfigSourceTest(unittest.TestCase):
    def test_repository_and_ref_are_split(self):
        self.assertEqual(
            configure_coverage_ci.parse_config_source("ROCm/rocm-libraries@develop"),
            ("ROCm/rocm-libraries", "develop"),
        )

    def test_empty_falls_back_to_the_default_source(self):
        self.assertEqual(
            configure_coverage_ci.parse_config_source(""),
            ("ROCm/rocm-libraries", "main"),
        )

    def test_missing_ref_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_coverage_ci.parse_config_source("ROCm/rocm-libraries")


class BuildCoverageMatrixTest(unittest.TestCase):
    def test_one_entry_per_project_and_family(self):
        matrix = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"],
            ["gfx94X-dcgpu", "gfx110X-all"],
            "ROCm/rocm-libraries",
            "main",
        )
        self.assertEqual(len(matrix), 2)
        self.assertEqual(
            [entry["amdgpu_families"] for entry in matrix],
            ["gfx94X-dcgpu", "gfx110X-all"],
        )

    def test_entries_are_json_serializable(self):
        matrix = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(json.loads(json.dumps(matrix)), matrix)

    def test_every_registered_project_declares_report_inputs(self):
        for name, project in configure_coverage_ci.COVERAGE_PROJECTS.items():
            with self.subTest(project=name):
                self.assertTrue(project.object_globs, "needs objects for llvm-cov")
                self.assertTrue(project.fetch_artifact_args, "needs artifacts to fetch")
                self.assertTrue(project.stage)
                self.assertTrue(project.test_component)

    def test_object_globs_reach_the_matrix(self):
        # The report job hands these globs to llvm-cov as they are, so they
        # have to survive into the matrix unchanged.
        (entry,) = configure_coverage_ci.build_coverage_matrix(
            ["hiprand"], ["gfx94X-dcgpu"], "ROCm/rocm-libraries", "main"
        )
        self.assertEqual(entry["object_globs"], "lib/libhiprand.so*")
        self.assertEqual(entry["fetch_artifact_args"], "--rand")


class BuildCoverageCmakeOptionsTest(unittest.TestCase):
    def test_full_selection_collapses_to_the_group_option(self):
        # What the default selection does: instrument every onboarded project.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(
                sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
            ),
            ["-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"],
        )

    def test_both_groups_collapse_to_the_combined_option(self):
        # No rocm-systems project is registered yet, so stand one in.
        with (
            mock.patch.dict(
                configure_coverage_ci.COVERAGE_PROJECTS,
                {"fakesystemslib": _fake_rocm_systems_project()},
            ),
            mock.patch.object(
                configure_coverage_ci,
                "ROCM_SYSTEMS_PROJECTS",
                frozenset({"fakesystemslib"}),
            ),
        ):
            self.assertEqual(
                configure_coverage_ci.build_coverage_cmake_options(
                    sorted(configure_coverage_ci.ROCM_LIBRARIES_PROJECTS)
                    + ["fakesystemslib"]
                ),
                ["-DTHEROCK_COVERAGE_ALL=ON"],
            )

    def test_partial_selection_names_each_project_in_upper_case(self):
        # therock_subproject.cmake only forwards the upper case spelling.
        self.assertEqual(
            configure_coverage_ci.build_coverage_cmake_options(["hipdnn"]),
            ["-DHIPDNN_ENABLE_COVERAGE=ON"],
        )

    def test_no_selection_produces_no_options(self):
        self.assertEqual(configure_coverage_ci.build_coverage_cmake_options([]), [])


class ResolveBuildStagesTest(unittest.TestCase):
    def test_math_libs_project_needs_the_math_libs_stage(self):
        self.assertEqual(
            configure_coverage_ci.resolve_build_stages(["hiprand"]),
            {"math-libs"},
        )

    def test_whole_rocm_libraries_group_stays_within_math_libs(self):
        keys = configure_coverage_ci.parse_projects("rocm_libraries_all")
        self.assertEqual(
            configure_coverage_ci.resolve_build_stages(keys), {"math-libs"}
        )


class RegistryMatchesBuildTopologyTest(unittest.TestCase):
    """The registry duplicates facts the build topology already knows.

    Nothing keeps the two in step at runtime, so a project moving between
    stages upstream would otherwise show up as a stage that is never built,
    hours into a run.
    """

    def test_every_project_names_the_stage_that_builds_its_artifacts(self):
        topology = get_topology()
        for key, project in sorted(configure_coverage_ci.COVERAGE_PROJECTS.items()):
            for artifact in project.artifact_names:
                with self.subTest(project=key, artifact=artifact):
                    self.assertEqual(
                        topology.get_stage_for_artifact(artifact),
                        project.stage,
                    )


class MainTest(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "github_output"
            output_path.touch()
            os.environ["GITHUB_OUTPUT"] = os.fspath(output_path)
            os.environ["PROJECTS_TO_TEST"] = "hiprand"
            os.environ["AMDGPU_FAMILIES"] = "gfx94X-dcgpu"
            os.environ["COVERAGE_CONFIG_SOURCE"] = "ROCm/rocm-libraries@main"

            self.assertEqual(configure_coverage_ci.main([]), 0)

            written = output_path.read_text()
            self.assertIn("coverage_matrix", written)
            self.assertIn("dist_amdgpu_families", written)
            self.assertIn("families_matrix_json", written)
            self.assertIn("coverage_cmake_options", written)
            # One project out of a populated group names that project alone.
            self.assertIn("-DHIPRAND_ENABLE_COVERAGE=ON", written)
            # The per-project inputs the report workflow reads back.
            self.assertIn('"test_component": "hiprand"', written)
            self.assertIn('"object_globs": "lib/libhiprand.so*"', written)
            # hipRAND is a math-libs project, so that stage has to be built.
            self.assertIn("needs_math_libs=true", written)


class EmitCmakeTest(unittest.TestCase):
    def test_emits_group_lists_from_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            text = out.read_text()
            libraries_line = next(
                line
                for line in text.splitlines()
                if line.startswith("set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS")
            )
            self.assertIn("hipRAND", libraries_line)
            self.assertIn("hipDNN", libraries_line)
            # No rocm-systems project is onboarded yet; CMakeLists.txt warns
            # when THEROCK_COVERAGE_ROCM_SYSTEMS_ALL is set with an empty list.
            self.assertIn("set(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS )\n", text)

    def test_emits_the_upstream_option_name_per_subproject(self):
        # The generic names are why this is a per-subproject map: setting
        # BUILD_CODE_COVERAGE globally would instrument everything that happens
        # to understand it.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "therock_coverage_projects.cmake"
            self.assertEqual(configure_coverage_ci.main(["--emit-cmake", str(out)]), 0)
            text = out.read_text()
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_HIPRAND BUILD_CODE_COVERAGE)", text
            )
            self.assertIn(
                "set(THEROCK_COVERAGE_OPTION_HIPDNN HIPDNN_ENABLE_COVERAGE)", text
            )

    def test_emit_cmake_needs_no_env(self):
        # Must work without PROJECTS_TO_TEST / AMDGPU_FAMILIES / GITHUB_OUTPUT set.
        saved = {
            k: os.environ.pop(k, None)
            for k in (
                "PROJECTS_TO_TEST",
                "AMDGPU_FAMILIES",
                "COVERAGE_CONFIG_SOURCE",
                "GITHUB_OUTPUT",
            )
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "c.cmake"
                self.assertEqual(
                    configure_coverage_ci.main(["--emit-cmake", str(out)]), 0
                )
                self.assertTrue(out.exists())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_emit_cmake_runs_without_site_packages(self):
        # CMakeLists.txt runs this on every configure, with whatever Python it
        # found, so it must not import anything outside the standard library.
        script = Path(configure_coverage_ci.__file__)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c.cmake"
            subprocess.run(
                [sys.executable, "-I", "-S", os.fspath(script), "--emit-cmake", out],
                check=True,
            )
            self.assertTrue(out.read_text().startswith("# Generated by"))


if __name__ == "__main__":
    unittest.main()
