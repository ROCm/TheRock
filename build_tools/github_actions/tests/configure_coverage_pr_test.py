# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import configure_coverage_ci
import configure_coverage_pr
from _therock_utils.build_topology import get_topology
from baseline_runs import RequiredArtifact

THEROCK = "ROCm/TheRock"
ROCM_LIBRARIES = "ROCm/rocm-libraries"


def _first(keys):
    return sorted(keys)[0] if keys else None


BLOCKED_PROJECT = _first(
    configure_coverage_ci.BLOCKED_PROJECTS & configure_coverage_ci.SUPPORTED_PROJECTS
)
UNSUPPORTED_PROJECT = _first(
    set(configure_coverage_ci.COVERAGE_PROJECTS)
    - configure_coverage_ci.SUPPORTED_PROJECTS
)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("unexpected call")


def _fake_run(run_id=4242):
    return SimpleNamespace(
        run_id=run_id,
        html_url=f"https://github.com/{THEROCK}/actions/runs/{run_id}",
    )


class ChangedProjectKeyTest(unittest.TestCase):
    def test_strips_projects_prefix_and_case(self):
        self.assertEqual(
            configure_coverage_pr.changed_project_key("projects/hipRAND"), "hiprand"
        )

    def test_nested_subtree_maps_to_its_project(self):
        self.assertEqual(
            configure_coverage_pr.changed_project_key("projects/hipblaslt/tensilelite"),
            "hipblaslt",
        )

    def test_bare_name_is_kept(self):
        self.assertEqual(
            configure_coverage_pr.changed_project_key(" rocrand "), "rocrand"
        )


class SelectProjectsTest(unittest.TestCase):
    def test_explicit_selection_wins_and_is_strict(self):
        selection = configure_coverage_pr.select_projects(
            "hiprand", "projects/rocrand", ["test:rocprim"]
        )
        self.assertEqual(selection.projects, ["hiprand"])
        self.assertEqual(selection.skipped, {})

        with self.assertRaises(ValueError):
            configure_coverage_pr.select_projects("not-a-project", "", [])

    def test_explicit_selection_rejects_stage_without_build_job(self):
        with self.assertRaises(ValueError):
            configure_coverage_pr.select_projects("rccl", "", [])

    def test_no_changed_projects_selects_nothing(self):
        selection = configure_coverage_pr.select_projects("", "", [])
        self.assertEqual(selection.projects, [])
        self.assertEqual(selection.skipped, {})

    def test_changed_projects_skip_what_cannot_be_measured(self):
        selection = configure_coverage_pr.select_projects(
            "", "projects/hiprand,projects/rocrand,shared/tensile", []
        )
        self.assertEqual(selection.projects, ["hiprand", "rocrand"])
        self.assertEqual(
            selection.skipped, {"shared/tensile": "not onboarded to coverage"}
        )

    def test_changed_blocked_project_is_skipped_with_reason(self):
        if BLOCKED_PROJECT is None:
            self.skipTest("no blocked coverage project registered")
        path = f"projects/{BLOCKED_PROJECT}"
        selection = configure_coverage_pr.select_projects("", path, [])
        self.assertEqual(selection.projects, [])
        self.assertTrue(selection.skipped[path].startswith("blocked: "))

    def test_changed_unsupported_project_is_skipped_with_reason(self):
        if UNSUPPORTED_PROJECT is None:
            self.skipTest("no unsupported coverage project registered")
        path = f"projects/{UNSUPPORTED_PROJECT}"
        selection = configure_coverage_pr.select_projects("", path, [])
        self.assertEqual(selection.projects, [])
        self.assertTrue(selection.skipped[path])

    def test_test_labels_without_changed_projects_select_nothing(self):
        selection = configure_coverage_pr.select_projects(
            "", "", ["test:hipRAND", "test_filter:quick"]
        )
        self.assertEqual(selection.projects, [])
        self.assertEqual(
            selection.skipped, {"test:hiprand": "not among the changed projects"}
        )

    def test_test_labels_intersect_changed_projects(self):
        selection = configure_coverage_pr.select_projects(
            "",
            "projects/hiprand,projects/rocrand",
            ["test:hiprand", "test:rocprim", "test:not-a-project"],
        )
        self.assertEqual(selection.projects, ["hiprand"])
        self.assertEqual(
            selection.skipped,
            {
                "test:not-a-project": "not onboarded to coverage",
                "test:rocprim": "not among the changed projects",
            },
        )

    def test_nothing_measurable_selects_nothing(self):
        selection = configure_coverage_pr.select_projects("", "shared/tensile", [])
        self.assertEqual(selection.projects, [])


class ResolveTestTypeTest(unittest.TestCase):
    def test_defaults_to_standard(self):
        self.assertEqual(configure_coverage_pr.resolve_test_type("", []), "standard")

    def test_label_overrides_input(self):
        self.assertEqual(
            configure_coverage_pr.resolve_test_type(
                "standard", ["test_filter:Comprehensive"]
            ),
            "comprehensive",
        )

    def test_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            configure_coverage_pr.resolve_test_type("fast", [])


class PlanBaselineTest(unittest.TestCase):
    def _plan(self, **overrides):
        kwargs = dict(
            baseline_run_id="",
            baseline_repository="",
            running_repository=THEROCK,
            therock_repository=THEROCK,
            find_baseline=_fail_if_called,
            changed_files=_fail_if_called,
        )
        kwargs.update(overrides)
        return configure_coverage_pr.plan_baseline(**kwargs)

    def test_explicit_baseline_from_this_repository_copies_compiler_runtime(self):
        plan = self._plan(baseline_run_id="123")
        self.assertEqual((plan.run_id, plan.repository), ("123", THEROCK))
        self.assertTrue(plan.explicit)
        self.assertTrue(plan.prebuilt_compiler_runtime)

    def test_explicit_baseline_from_another_repository_builds_compiler_runtime(self):
        plan = self._plan(
            baseline_run_id="123",
            baseline_repository=THEROCK,
            running_repository=ROCM_LIBRARIES,
        )
        self.assertEqual(plan.repository, THEROCK)
        self.assertFalse(plan.prebuilt_compiler_runtime)
        self.assertIn(THEROCK, plan.notes[0])

    def test_no_baseline_found_falls_back_to_self_contained(self):
        plan = self._plan(find_baseline=lambda: None)
        self.assertEqual(plan.run_id, "")
        self.assertFalse(plan.prebuilt_compiler_runtime)
        self.assertTrue(plan.notes)

    def test_found_baseline_copies_compiler_runtime_when_unaffected(self):
        plan = self._plan(
            find_baseline=lambda: _fake_run(),
            changed_files=lambda: ["rocm-libraries"],
        )
        self.assertEqual((plan.run_id, plan.repository), ("4242", THEROCK))
        self.assertTrue(plan.html_url.endswith("/4242"))
        self.assertTrue(plan.prebuilt_compiler_runtime)

    def test_found_baseline_rebuilds_compiler_runtime_when_affected(self):
        plan = self._plan(
            find_baseline=lambda: _fake_run(),
            changed_files=lambda: ["cmake/therock_subproject.cmake"],
        )
        self.assertEqual(plan.run_id, "4242")
        self.assertFalse(plan.prebuilt_compiler_runtime)
        self.assertTrue(plan.notes[0].startswith("compiler-runtime is built: "))

    def test_rebuild_note_names_one_reason_and_counts_the_rest(self):
        plan = self._plan(
            find_baseline=lambda: _fake_run(),
            changed_files=lambda: [f"build_tools/x{i}.py" for i in range(3)],
        )
        self.assertIn("build_tools/x0.py", plan.notes[0])
        self.assertTrue(plan.notes[0].endswith(" (and 2 more)."))

    def test_found_baseline_rebuilds_compiler_runtime_without_changed_files(self):
        plan = self._plan(find_baseline=lambda: _fake_run(), changed_files=lambda: None)
        self.assertFalse(plan.prebuilt_compiler_runtime)

    def test_cross_repository_baseline_never_reads_changed_files(self):
        plan = self._plan(
            running_repository=ROCM_LIBRARIES,
            find_baseline=lambda: _fake_run(),
        )
        self.assertEqual(plan.repository, THEROCK)
        self.assertFalse(plan.prebuilt_compiler_runtime)


class PrChangedFilesTest(unittest.TestCase):
    def test_only_pull_requests_have_changed_files(self):
        with mock.patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "workflow_dispatch"}
        ), mock.patch.object(
            configure_coverage_pr, "get_git_modified_paths", side_effect=_fail_if_called
        ):
            self.assertIsNone(configure_coverage_pr._pr_changed_files())

    def test_diffs_against_the_merge_commit_parent(self):
        with mock.patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ), mock.patch.object(
            configure_coverage_pr,
            "get_git_modified_paths",
            return_value=iter(["rocm-libraries"]),
        ) as diff:
            self.assertEqual(
                configure_coverage_pr._pr_changed_files(), ["rocm-libraries"]
            )
        diff.assert_called_once_with("HEAD^1")

    def test_failed_diff_counts_as_unknown_changes(self):
        error = subprocess.CalledProcessError(128, ["git", "diff"])
        with mock.patch.dict(
            os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
        ), mock.patch.object(
            configure_coverage_pr, "get_git_modified_paths", side_effect=error
        ):
            self.assertIsNone(configure_coverage_pr._pr_changed_files())


class RebuildArtifactsTest(unittest.TestCase):
    def test_grouped_projects_share_one_artifact(self):
        self.assertEqual(
            configure_coverage_pr.rebuild_artifacts_for(["hiprand", "rocrand"]),
            ["rand"],
        )

    def test_every_default_project_artifact_is_built_by_its_stage(self):
        # configure_stage.py --artifacts only builds what the stage produces, so
        # a stale name would silently build nothing for the project.
        topology = get_topology()
        artifacts_by_group = topology.get_artifact_group_to_artifacts()
        for key in sorted(configure_coverage_ci.DEFAULT_PROJECTS):
            project = configure_coverage_ci.COVERAGE_PROJECTS[key]
            produced = {
                name
                for group in topology.build_stages[project.stage].artifact_groups
                for name in artifacts_by_group.get(group, [])
            }
            with self.subTest(project=key):
                self.assertTrue(project.artifact_names)
                self.assertLessEqual(set(project.artifact_names), produced)


class FindBaselineRunTest(unittest.TestCase):
    def test_requires_the_selected_artifacts_for_every_target(self):
        self.assertEqual(
            set(
                configure_coverage_pr.baseline_artifacts(["hiprand"], ["gfx94X-dcgpu"])
            ),
            {
                RequiredArtifact(name="rand", target_family="generic"),
                RequiredArtifact(name="rand", target_family="gfx942"),
            },
        )

    def test_requires_only_the_linux_stage_builds_it_uses(self):
        jobs = configure_coverage_pr.baseline_build_jobs(["gfx94X-dcgpu"])
        names = [
            "Linux::release / Build Multi-Arch Stages / compiler-runtime / Stage - Compiler Runtime",
            "Linux::release / Build Multi-Arch Stages / math-libs (gfx94X-dcgpu, gfx942, linux-gfx942-1gpu, ... / Stage - Math Libs (gfx94X-dcgpu)",
        ]
        unrelated = [
            "Windows::release / Build Multi-Arch Stages / compiler-runtime / Stage - Compiler Runtime",
            "Linux::release / Build Multi-Arch Stages / math-libs (gfx950-dcgpu, gfx950, ... / Stage - Math Libs (gfx950-dcgpu)",
            "Linux::release / Build DEB Packages / Build deb packages",
        ]
        for job in jobs:
            with self.subTest(job=job):
                self.assertEqual(sum(job in name for name in names), 1)
                self.assertFalse(any(job in name for name in unrelated))

    def test_passes_the_requirements_to_select_baseline_run(self):
        with mock.patch.object(
            configure_coverage_pr, "select_baseline_run", return_value=None
        ) as select:
            configure_coverage_pr.find_baseline_run(
                ["hiprand"],
                ["gfx94X-dcgpu"],
                github_repository=THEROCK,
                max_age_hours=72.0,
            )
        kwargs = select.call_args.kwargs
        self.assertEqual(kwargs["github_repository"], THEROCK)
        self.assertEqual(kwargs["platform"], "linux")
        self.assertEqual(kwargs["max_age_hours"], 72.0)
        self.assertEqual(kwargs["max_runs"], configure_coverage_pr.BASELINE_MAX_RUNS)
        self.assertEqual(
            kwargs["required_successful_job_name_substrings"],
            configure_coverage_pr.baseline_build_jobs(["gfx94X-dcgpu"]),
        )
        self.assertEqual(
            set(kwargs["required_artifacts"]),
            set(
                configure_coverage_pr.baseline_artifacts(["hiprand"], ["gfx94X-dcgpu"])
            ),
        )


class MainTest(unittest.TestCase):
    def _run(self, env, event=None):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            summary = Path(tmp) / "summary"
            full_env = {
                "GITHUB_OUTPUT": os.fspath(output),
                "GITHUB_STEP_SUMMARY": os.fspath(summary),
                "GITHUB_REPOSITORY": THEROCK,
                **env,
            }
            if event is not None:
                event_path = Path(tmp) / "event.json"
                event_path.write_text(json.dumps(event))
                full_env["GITHUB_EVENT_PATH"] = os.fspath(event_path)
            with mock.patch.dict(os.environ, full_env, clear=True):
                self.assertEqual(configure_coverage_pr.main([]), 0)
            outputs = dict(
                line.split("=", 1) for line in output.read_text().splitlines()
            )
            return outputs, summary.read_text()

    def test_explicit_baseline_narrows_the_build(self):
        outputs, summary = self._run(
            {
                "CHANGED_PROJECTS": "projects/hiprand",
                "BASELINE_RUN_ID": "123",
            }
        )
        self.assertEqual(
            outputs,
            {
                "enabled": "true",
                "projects_to_test": "hiprand",
                "amdgpu_families": "gfx94X-dcgpu",
                "test_type": "standard",
                "baseline_run_id": "123",
                "baseline_repository": THEROCK,
                "rebuild_artifacts": "rand",
                "prebuilt_compiler_runtime": "true",
            },
        )
        self.assertIn("`hiprand` | yes", summary)

    def test_without_baseline_builds_whole_stage(self):
        with mock.patch.object(
            configure_coverage_pr, "find_baseline_run", return_value=None
        ):
            outputs, _ = self._run({"PROJECTS_TO_TEST": "hiprand"})
        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(outputs["baseline_run_id"], "")
        self.assertEqual(outputs["rebuild_artifacts"], "")
        self.assertEqual(outputs["prebuilt_compiler_runtime"], "false")

    def test_failed_lookup_degrades_to_no_baseline(self):
        with mock.patch.object(
            configure_coverage_pr,
            "find_baseline_run",
            side_effect=RuntimeError("rate limited"),
        ):
            outputs, _ = self._run({"PROJECTS_TO_TEST": "hiprand"})
        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(outputs["baseline_run_id"], "")

    def test_pull_request_labels_select_projects_and_test_type(self):
        event = {
            "pull_request": {
                "labels": [
                    {"name": "test:hiprand"},
                    {"name": "test_filter:quick"},
                ]
            }
        }
        with mock.patch.object(
            configure_coverage_pr, "find_baseline_run", return_value=_fake_run()
        ), mock.patch.object(
            configure_coverage_pr, "_pr_changed_files", return_value=["rocm-libraries"]
        ):
            outputs, summary = self._run(
                {
                    "GITHUB_EVENT_NAME": "pull_request",
                    "CHANGED_PROJECTS": "projects/hiprand,projects/rocrand",
                },
                event,
            )
        self.assertEqual(outputs["projects_to_test"], "hiprand")
        self.assertEqual(outputs["test_type"], "quick")
        self.assertEqual(outputs["baseline_run_id"], "4242")
        self.assertEqual(outputs["prebuilt_compiler_runtime"], "true")
        self.assertIn("compiler-runtime: copied from the baseline.", summary)

    def test_empty_selection_disables_the_run_without_a_lookup(self):
        with mock.patch.object(
            configure_coverage_pr, "find_baseline_run", side_effect=_fail_if_called
        ):
            outputs, summary = self._run({"CHANGED_PROJECTS": "shared/tensile"})
        self.assertEqual(outputs["enabled"], "false")
        self.assertEqual(outputs["projects_to_test"], "")
        self.assertIn("nothing is built", summary)

    def test_pull_request_without_changed_projects_measures_nothing(self):
        event = {"pull_request": {"labels": [{"name": "test:hiprand"}]}}
        with mock.patch.object(
            configure_coverage_pr, "find_baseline_run", side_effect=_fail_if_called
        ):
            outputs, summary = self._run({"GITHUB_EVENT_NAME": "pull_request"}, event)
        self.assertEqual(outputs["enabled"], "false")
        self.assertEqual(outputs["projects_to_test"], "")
        self.assertIn("`test:hiprand` | no: not among the changed projects", summary)


if __name__ == "__main__":
    unittest.main()
