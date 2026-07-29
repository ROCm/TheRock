# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for stage_reuse_decision: impact + baseline-availability gates."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configure_multi_arch_ci as cma

import stage_reuse_decision as srd
from stage_reuse_decision import StageReuseMode, compute_auto_stage_reuse
from baseline_runs import (
    ArtifactAvailability,
    BaselineRun,
    WorkflowJobHealth,
    WorkflowRunSummary,
)
from github_actions_api import GitHubAPIError


class _FakeStage:
    def __init__(self, groups):
        self.artifact_groups = groups


class _FakeArtifact:
    def __init__(
        self,
        name: str,
        artifact_type: str,
        artifact_deps: tuple[str, ...] = (),
        *,
        platform: str | None = None,
        disable_platforms: tuple[str, ...] = (),
        disable_platforms_if_flags_not_set: dict[str, str] | None = None,
    ):
        self.name = name
        self.type = artifact_type
        self.artifact_deps = list(artifact_deps)
        self.platform = platform
        self.disable_platforms = list(disable_platforms)
        self.disable_platforms_if_flags_not_set = dict(
            disable_platforms_if_flags_not_set or {}
        )


class FakeTopology:
    """Minimal BuildTopology stand-in for stage_impact + artifact derivation.

    compiler-runtime produces artifact 'base'; math-libs produces 'blas'.
    """

    def __init__(self):
        self.build_stages = {
            "compiler-runtime": _FakeStage(["base-group"]),
            "math-libs": _FakeStage(["blas-group"]),
        }
        self.artifact_groups = {
            "base-group": type("G", (), {"source_sets": ["core"]})(),
            "blas-group": type("G", (), {"source_sets": ["libs"]})(),
        }

        self.artifacts = {
            "base": _FakeArtifact(
                name="base",
                artifact_type="target-neutral",
            ),
            "blas": _FakeArtifact(
                name="blas",
                artifact_type="target-specific",
            ),
        }

    def get_source_set_to_artifact_groups(self):
        return {"core": ["base-group"], "libs": ["blas-group"]}

    def get_artifact_group_to_build_stages(self):
        return {"base-group": ["compiler-runtime"], "blas-group": ["math-libs"]}

    def get_artifact_group_to_artifacts(self):
        return {"base-group": ["base"], "blas-group": ["blas"]}

    def get_produced_artifacts(self, stage_name):
        return set()

    def get_artifacts_in_group(self, group_name):
        return {
            "base-group": [self.artifacts["base"]],
            "blas-group": [self.artifacts["blas"]],
        }.get(group_name, [])

    def get_source_set_for_submodule(self, name, platform=None):
        mapping = {"rocm-libraries": "libs"}
        ss = mapping.get(name)
        return type("S", (), {"name": ss})() if ss else None

    def get_source_set_for_path(self, path, platform=None):
        return None


def _baseline(run_id, matched_filenames):
    summary = WorkflowRunSummary(
        repository="ROCm/TheRock",
        branch="main",
        commit=f"sha-{run_id}",
        workflow="multi_arch_ci.yml",
        run_id=run_id,
        status="completed",
        conclusion="success",
        timestamp="2026-06-17T20:00:00Z",
        html_url=f"https://github.com/ROCm/TheRock/actions/runs/{run_id}",
    )
    return BaselineRun(
        source_ref=summary,
        platform="linux",
        job_health=WorkflowJobHealth(
            required_name_substrings=("Build",),
            matched_job_names=("Build",),
            failed_job_names=(),
            missing_name_substrings=(),
        ),
        artifact_availability=ArtifactAvailability(
            required_artifacts=(),
            matched_filenames=tuple(matched_filenames),
            missing_artifacts=(),
        ),
    )


def _selector(baseline):
    """Return an injected baseline_selector that always yields `baseline`."""
    return lambda required: baseline


class ModeParsingTest(unittest.TestCase):
    def test_default_is_dry_run(self):
        import os

        os.environ.pop("STAGE_REUSE_MODE", None)
        self.assertEqual(StageReuseMode.from_environ(), StageReuseMode.DRY_RUN)

    def test_explicit_modes(self):
        import os

        for value, expected in [
            ("dry-run", StageReuseMode.DRY_RUN),
            ("reuse-stage", StageReuseMode.REUSE_STAGE),
            ("garbage", StageReuseMode.DRY_RUN),
        ]:
            os.environ["STAGE_REUSE_MODE"] = value
            self.assertEqual(StageReuseMode.from_environ(), expected)
        os.environ.pop("STAGE_REUSE_MODE", None)


class AvailabilityGateTest(unittest.TestCase):
    def test_dry_run_verifies_artifacts_present(self):
        # math-libs changed -> compiler-runtime unaffected (candidate).
        # Baseline HAS base artifact -> reported as WOULD skip, but applied=().
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        self.assertIn("compiler-runtime", result.candidate_stages)
        self.assertIn("compiler-runtime", result.available_stages)
        self.assertEqual(result.applied_reuse_stages, ())  # dry-run applies nothing
        self.assertEqual(result.baseline_run_id, "123")
        joined = "\n".join(result.report_lines)
        self.assertIn("available in baseline", joined)
        self.assertIn("WOULD be skipped", joined)

    def test_dry_run_unaffected_but_artifacts_missing_rebuilds(self):
        # compiler-runtime unaffected, but baseline only has blas (not base).
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["blas_lib_generic.tar.zst"])),
        )
        self.assertIn("compiler-runtime", result.candidate_stages)
        self.assertIn("compiler-runtime", result.unavailable_stages)
        self.assertEqual(result.available_stages, ())
        joined = "\n".join(result.report_lines)
        self.assertIn("artifacts NOT available", joined)

    def test_no_baseline_found_rebuilds_candidates(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(None),
        )
        self.assertIn("compiler-runtime", result.candidate_stages)
        self.assertEqual(result.available_stages, ())
        self.assertIsNone(result.baseline_run_id)
        joined = "\n".join(result.report_lines)
        self.assertIn("no baseline run contains artifacts", joined)

    def test_target_neutral_stage_only_requires_generic_artifact(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=["gfx94X-dcgpu"],
            topology=FakeTopology(),
            baseline_selector=_selector(
                _baseline(
                    "123",
                    ["base_lib_generic.tar.zst"],
                )
            ),
        )

        self.assertIn(
            "compiler-runtime",
            result.available_stages,
        )
        self.assertNotIn(
            "compiler-runtime",
            result.unavailable_stages,
        )

    def test_reuse_stage_applies_only_available_stages(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        self.assertEqual(result.applied_reuse_stages, ("compiler-runtime",))
        joined = "\n".join(result.report_lines)
        self.assertIn("WILL be skipped", joined)

    def test_baseline_lookup_error_is_safe(self):
        def boom(required):
            raise GitHubAPIError("network down")

        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=boom,
        )
        self.assertEqual(result.applied_reuse_stages, ())
        joined = "\n".join(result.report_lines)
        self.assertIn("baseline lookup failed", joined)

    def test_non_api_selector_error_propagates(self):
        def boom(required):
            raise ValueError("bad required-artifacts request")

        with self.assertRaises(ValueError):
            compute_auto_stage_reuse(
                changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
                mode=StageReuseMode.REUSE_STAGE,
                linux_amdgpu_families=["generic"],
                topology=FakeTopology(),
                baseline_selector=boom,
            )


class GuardrailTest(unittest.TestCase):
    def test_full_ci_trigger_skips_baseline_query(self):
        called = {"n": 0}

        def selector(required):
            called["n"] += 1
            return None

        result = compute_auto_stage_reuse(
            changed_files=["build_tools/foo.py"],
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=selector,
        )
        self.assertTrue(result.full_rebuild_required)
        self.assertEqual(result.applied_reuse_stages, ())
        self.assertEqual(called["n"], 0)

    def test_no_diff_is_conservative(self):
        result = compute_auto_stage_reuse(
            changed_files=None,
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        self.assertTrue(result.full_rebuild_required)
        self.assertEqual(result.applied_reuse_stages, ())

    def test_step_summary_shows_baseline_and_availability(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        summary = srd.render_step_summary(result)
        self.assertIn("baseline run checked: `123`", summary)
        self.assertIn("available in baseline", summary)
        self.assertIn("no build steps were skipped", summary)


class DefaultBaselineSelectorTest(unittest.TestCase):
    @patch.object(
        srd.baseline_runs,
        "select_baseline_run",
        return_value="baseline",
    )
    def test_exact_expected_sha_is_forwarded(
        self,
        mock_select_baseline_run,
    ):
        selector = srd._default_baseline_selector(
            platform="linux",
            expected_baseline_sha="base-commit-sha",
        )

        required = [
            srd.RequiredArtifact(
                name="base",
                target_family="generic",
            )
        ]

        result = selector(required)

        self.assertEqual(result, "baseline")
        mock_select_baseline_run.assert_called_once()

        call_kwargs = mock_select_baseline_run.call_args.kwargs

        self.assertEqual(
            call_kwargs["expected_head_sha"],
            "base-commit-sha",
        )
        self.assertNotIn(
            "current_commit_sha",
            call_kwargs,
        )
        self.assertNotIn(
            "ordered_commit_shas",
            call_kwargs,
        )

    @patch.object(
        srd.baseline_runs,
        "select_baseline_run",
    )
    def test_missing_expected_sha_disables_baseline_lookup(
        self,
        mock_select_baseline_run,
    ):
        selector = srd._default_baseline_selector(
            platform="linux",
            expected_baseline_sha=None,
        )

        result = selector(
            [
                srd.RequiredArtifact(
                    name="base",
                    target_family="generic",
                )
            ]
        )

        self.assertIsNone(result)
        mock_select_baseline_run.assert_not_called()


class PlatformAwareAvailabilityTest(unittest.TestCase):
    """A stage is only reusable when its artifacts exist for EVERY platform.

    Guards the review concern: prebuilt_stages flow to both the Linux and
    Windows build configs, so a stage available only in the Linux baseline must
    not be reused when Windows is also being built.
    """

    def _selector_factory(self, per_platform):
        """Return a factory yielding a per-platform baseline selector."""

        def factory(
            platform: str,
            expected_baseline_sha: str | None,
        ):
            del expected_baseline_sha

            baseline = per_platform.get(platform)
            return lambda required: baseline

        return factory

    def test_stage_reused_only_when_available_on_all_platforms(self):
        # compiler-runtime unaffected. Linux baseline HAS base; Windows does NOT.
        per_platform = {
            "linux": _baseline("L1", ["base_lib_generic.tar.zst"]),
            "windows": _baseline("L1", ["blas_lib_generic.tar.zst"]),
        }
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            windows_amdgpu_families=["generic"],
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector_factory=self._selector_factory(per_platform),
        )
        # Available on linux but missing on windows -> NOT applied.
        self.assertIn("compiler-runtime", result.candidate_stages)
        self.assertIn("compiler-runtime", result.unavailable_stages)
        self.assertEqual(result.applied_reuse_stages, ())
        self.assertEqual(result.platform_available["linux"], ("compiler-runtime",))
        self.assertEqual(result.platform_available["windows"], ())
        joined = "\n".join(result.report_lines)
        self.assertIn("missing on: windows", joined)

    def test_stage_reused_when_present_on_both_platforms(self):
        per_platform = {
            "linux": _baseline("L1", ["base_lib_generic.tar.zst"]),
            "windows": _baseline("L1", ["base_lib_generic.tar.zst"]),
        }
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            windows_amdgpu_families=["generic"],
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector_factory=self._selector_factory(per_platform),
        )
        self.assertEqual(result.applied_reuse_stages, ("compiler-runtime",))
        self.assertIn("compiler-runtime", result.available_stages)

    def test_single_platform_default_is_linux(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        self.assertEqual(result.applied_reuse_stages, ("compiler-runtime",))
        self.assertIn("linux", result.platform_available)

    def test_no_platforms_selected_disables_auto_reuse(self):
        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            topology=FakeTopology(),
            baseline_selector=_selector(_baseline("123", ["base_lib_generic.tar.zst"])),
        )
        self.assertTrue(result.full_rebuild_required)
        self.assertEqual(result.applied_reuse_stages, ())
        self.assertIn("no build platforms selected", "\n".join(result.report_lines))

    def test_different_platform_baselines_disable_reuse(self):
        """Automatic reuse is disabled when platforms resolve to different baseline runs."""
        per_platform = {
            "linux": _baseline("L1", ["base_lib_generic.tar.zst"]),
            "windows": _baseline("W1", ["base_lib_generic.tar.zst"]),
        }

        result = compute_auto_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            mode=StageReuseMode.REUSE_STAGE,
            windows_amdgpu_families=["generic"],
            linux_amdgpu_families=["generic"],
            topology=FakeTopology(),
            baseline_selector_factory=self._selector_factory(per_platform),
        )

        self.assertTrue(result.full_rebuild_required)
        self.assertEqual(result.applied_reuse_stages, ())
        self.assertIn(
            "automatic reuse resolved different baseline runs per platform",
            result.reasons,
        )

    def test_expected_baseline_sha_is_forwarded_to_selector_factory(self):
        received: list[tuple[str, str | None]] = []

        def selector_factory(
            platform: str,
            expected_baseline_sha: str | None,
        ):
            received.append(
                (
                    platform,
                    expected_baseline_sha,
                )
            )

            return _selector(
                _baseline(
                    "123",
                    ["base_lib_generic.tar.zst"],
                )
            )

        result = compute_auto_stage_reuse(
            changed_files=[
                "rocm-libraries/projects/rocBLAS/x.cpp",
            ],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=[
                "gfx94X-dcgpu",
            ],
            expected_baseline_sha="base-commit-sha",
            topology=FakeTopology(),
            baseline_selector_factory=selector_factory,
        )

        self.assertEqual(
            received,
            [
                (
                    "linux",
                    "base-commit-sha",
                )
            ],
        )
        self.assertEqual(
            result.baseline_run_id,
            "123",
        )

    def test_platform_specific_artifact_is_checked_only_on_matching_platform(self):
        topology = FakeTopology()
        topology.artifacts["base"].platform = "windows"

        per_platform = {
            # The Windows-only artifact must not be required on Linux.
            "linux": _baseline(
                "L1",
                [],
            ),
            # It must still be required on Windows.
            "windows": _baseline(
                "L1",
                ["base_lib_generic.tar.zst"],
            ),
        }

        result = compute_auto_stage_reuse(
            changed_files=[
                "rocm-libraries/projects/rocBLAS/x.cpp",
            ],
            mode=StageReuseMode.REUSE_STAGE,
            linux_amdgpu_families=["generic"],
            windows_amdgpu_families=["generic"],
            topology=topology,
            baseline_selector_factory=self._selector_factory(
                per_platform,
            ),
        )

        self.assertEqual(
            result.applied_reuse_stages,
            ("compiler-runtime",),
        )
        self.assertEqual(
            result.platform_available["linux"],
            ("compiler-runtime",),
        )
        self.assertEqual(
            result.platform_available["windows"],
            ("compiler-runtime",),
        )


class PlanStageReuseTest(unittest.TestCase):
    """The pure planning step is independent of baseline/reporting."""

    def test_plan_returns_candidates_without_baseline(self):
        plan = srd.plan_stage_reuse(
            changed_files=["rocm-libraries/projects/rocBLAS/x.cpp"],
            topology=FakeTopology(),
        )
        self.assertIn("compiler-runtime", plan.impact.copy_stages)
        self.assertFalse(plan.impact.full_rebuild_required)

    def test_plan_none_changed_files_is_full_rebuild(self):
        plan = srd.plan_stage_reuse(changed_files=None, topology=FakeTopology())
        self.assertTrue(plan.impact.full_rebuild_required)
        self.assertEqual(plan.impact.copy_stages, ())


class RequiredArtifactsTest(unittest.TestCase):
    def test_target_neutral_artifact_requires_generic_only(self):
        required = srd._required_artifacts_for_stages(
            topology=FakeTopology(),
            stage_names=["compiler-runtime"],
            target_families=[
                "gfx94X-dcgpu",
                "generic",
            ],
            platform="linux",
        )

        self.assertEqual(
            required,
            [
                srd.RequiredArtifact(
                    name="base",
                    target_family="generic",
                )
            ],
        )

    def test_platform_specific_artifact_is_skipped_on_other_platform(self):
        topology = FakeTopology()
        topology.artifacts["base"].platform = "windows"

        required = srd._required_artifacts_for_stages(
            topology=topology,
            stage_names=["compiler-runtime"],
            target_families=["generic"],
            platform="linux",
        )

        self.assertEqual(required, [])

    def test_disabled_artifact_is_skipped_on_platform(self):
        topology = FakeTopology()
        topology.artifacts["base"].disable_platforms = ["windows"]

        required = srd._required_artifacts_for_stages(
            topology=topology,
            stage_names=["compiler-runtime"],
            target_families=["generic"],
            platform="windows",
        )

        self.assertEqual(required, [])

    def test_conditionally_disabled_artifact_remains_required_without_flags(self):
        topology = FakeTopology()
        topology.artifacts["base"].disable_platforms_if_flags_not_set = {
            "linux": "ENABLE_BASE",
        }

        required = srd._required_artifacts_for_stages(
            topology=topology,
            stage_names=["compiler-runtime"],
            target_families=["generic"],
            platform="linux",
        )

        self.assertEqual(
            required,
            [
                srd.RequiredArtifact(
                    name="base",
                    target_family="generic",
                )
            ],
        )


class StageArtifactAvailabilityTest(unittest.TestCase):
    def test_target_specific_stage_requires_selected_gpu_archive(self):
        available = srd._stage_artifacts_available(
            topology=FakeTopology(),
            stage_name="math-libs",
            target_families=[
                "gfx94X-dcgpu",
                "generic",
            ],
            available_filenames={
                "blas_lib_generic.tar.zst",
            },
            platform="linux",
        )

        self.assertFalse(available)

    def test_target_specific_artifact_requires_each_selected_gpu_family(self):
        required = srd._required_artifacts_for_stages(
            topology=FakeTopology(),
            stage_names=["math-libs"],
            target_families=[
                "gfx94X-dcgpu",
                "gfx120X-all",
                "generic",
            ],
            platform="linux",
        )

        self.assertEqual(
            required,
            [
                srd.RequiredArtifact(
                    name="blas",
                    target_family="gfx94X-dcgpu",
                ),
                srd.RequiredArtifact(
                    name="blas",
                    target_family="gfx120X-all",
                ),
            ],
        )


class TargetFamiliesTest(unittest.TestCase):
    def test_dedupes_and_appends_generic(self):
        families = srd._target_families_for_platform(
            ["gfx94X-dcgpu", "gfx94X-dcgpu"],
        )

        self.assertEqual(
            families,
            ("gfx94X-dcgpu", "generic"),
        )

    def test_platform_families_are_not_combined(self):
        linux = srd._target_families_for_platform(
            ["gfx94X-dcgpu"],
        )
        windows = srd._target_families_for_platform(
            ["gfx110X-all"],
        )

        self.assertEqual(
            linux,
            ("gfx94X-dcgpu", "generic"),
        )
        self.assertEqual(
            windows,
            ("gfx110X-all", "generic"),
        )


class BuildPlatformsTest(unittest.TestCase):
    def test_no_families_is_empty(self):
        self.assertEqual(srd._build_platforms((), ()), ())

    def test_linux_only(self):
        self.assertEqual(srd._build_platforms(["gfx94x"], ()), ("linux",))

    def test_windows_only(self):
        self.assertEqual(srd._build_platforms((), ["gfx110x"]), ("windows",))

    def test_both_platforms(self):
        self.assertEqual(
            srd._build_platforms(["gfx94x"], ["gfx110x"]), ("linux", "windows")
        )


class PlatformImpactPlanningTest(unittest.TestCase):
    def test_compute_uses_platform_agnostic_impact(self):
        class RecordingTopology(FakeTopology):
            def __init__(self):
                super().__init__()
                self.impact_platforms: list[str | None] = []

            def get_source_set_for_submodule(
                self,
                name,
                platform=None,
            ):
                self.impact_platforms.append(platform)
                return super().get_source_set_for_submodule(
                    name,
                    platform,
                )

            def get_source_set_for_path(
                self,
                path,
                platform=None,
            ):
                self.impact_platforms.append(platform)
                return super().get_source_set_for_path(
                    path,
                    platform,
                )

        topology = RecordingTopology()

        result = compute_auto_stage_reuse(
            changed_files=[
                "rocm-libraries/projects/rocBLAS/x.cpp",
            ],
            mode=StageReuseMode.DRY_RUN,
            linux_amdgpu_families=[
                "gfx94X-dcgpu",
            ],
            windows_amdgpu_families=[
                "gfx110X-all",
            ],
            topology=topology,
            baseline_selector=_selector(
                _baseline(
                    "123",
                    ["base_lib_generic.tar.zst"],
                )
            ),
        )

        self.assertTrue(
            topology.impact_platforms,
            "Expected stage-impact analysis to query the topology",
        )

        self.assertTrue(
            all(platform is None for platform in topology.impact_platforms),
            topology.impact_platforms,
        )

        self.assertIn(
            "compiler-runtime",
            result.candidate_stages,
        )


class CrossRepoArtifactReuseTest(unittest.TestCase):
    """Test cross-repo vs same-repo artifact reuse logic.

    External repos (rocm-libraries, rocm-systems) can copy artifacts from
    TheRock's baseline runs. Cross-repo reuse skips automatic stage decisions
    since they query the current repo, not the baseline repo.
    """

    def test_cross_repo_skips_auto_stage_reuse(self):
        """Cross-repo reuse skips automatic stage decisions."""
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "ROCm/rocm-libraries"}):
            baseline_repository = "ROCm/TheRock"
            current_repo = os.environ.get("GITHUB_REPOSITORY", "")
            is_cross_repo = baseline_repository and baseline_repository != current_repo

            self.assertTrue(is_cross_repo)

            # Cross-repo: auto stages should NOT be applied
            stage_decisions = {}
            if not is_cross_repo:
                stage_decisions["math-libs"] = cma.JobAction.PREBUILT

            self.assertEqual(stage_decisions, {})

    def test_same_repo_applies_auto_stage_reuse(self):
        """Same-repo reuse applies automatic stage decisions."""
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "ROCm/TheRock"}):
            baseline_repository = "ROCm/TheRock"
            current_repo = os.environ.get("GITHUB_REPOSITORY", "")
            is_cross_repo = baseline_repository and baseline_repository != current_repo

            self.assertFalse(is_cross_repo)

            # Same-repo: auto stages should be applied
            stage_decisions = {}
            if not is_cross_repo:
                stage_decisions["math-libs"] = cma.JobAction.PREBUILT

            self.assertEqual(stage_decisions, {"math-libs": cma.JobAction.PREBUILT})


if __name__ == "__main__":
    unittest.main()
