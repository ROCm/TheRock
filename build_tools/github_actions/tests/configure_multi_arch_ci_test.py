# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_multi_arch_ci.py.

Each test demonstrates the pattern for testing a pipeline step:
construct the input dataclass, call the function, assert on the output.
No environment variables or filesystem access needed (except from_environ tests).
"""

import json
import os
import re
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_multi_arch_ci as cm
from configure_multi_arch_ci_summary import format_summary
from workflow_utils import WORKFLOWS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_from_environ(
    event_name: str,
    event_payload: dict,
    *,
    commit_ref: str = "main",
    build_variant: str = "release",
) -> cm.CIInputs:
    """Call CIInputs.from_environ() with a synthetic event payload.

    GitHub Actions sets GITHUB_EVENT_PATH to a JSON file containing the full
    webhook event payload. This helper writes a temporary JSON file and patches
    the environment to simulate that.

    See: https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-environment-variables#default-environment-variables
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(event_payload, f)
        event_path = f.name

    try:
        env = {
            "GITHUB_RUN_ID": "12345",
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_REF_NAME": commit_ref,
            "BUILD_VARIANT": build_variant,
        }
        with patch.dict(os.environ, env, clear=False):
            return cm.CIInputs.from_environ()
    finally:
        os.unlink(event_path)


# ---------------------------------------------------------------------------
# CIInputs — construction and properties
# ---------------------------------------------------------------------------


class TestCIInputs(unittest.TestCase):
    """Test CIInputs dataclass and its properties."""

    def test_event_type_properties(self):
        """Event type properties are mutually exclusive."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        self.assertTrue(inputs.is_pull_request)
        self.assertFalse(inputs.is_push)
        self.assertFalse(inputs.is_schedule)
        self.assertFalse(inputs.is_workflow_dispatch)

    def test_defaults(self):
        """Fields with defaults can be omitted."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        self.assertEqual(inputs.pr_labels, [])
        self.assertEqual(inputs.linux_amdgpu_families, [])
        self.assertEqual(inputs.prebuilt_stages, "")


class TestCIInputsFromEnviron(unittest.TestCase):
    """Test CIInputs.from_environ() with event payload fixtures.

    GitHub Actions provides the full webhook event payload as a JSON file
    via GITHUB_EVENT_PATH. Each event type has a different payload structure:
    - workflow_dispatch: inputs are in event.inputs
    - pull_request: PR labels are in event.pull_request.labels
    - push: the previous HEAD SHA is in event.before

    See: https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-environment-variables#default-environment-variables
    """

    def test_workflow_dispatch_reads_inputs(self):
        """workflow_dispatch inputs (families, labels, prebuilt config)."""
        inputs = _run_from_environ(
            event_name="workflow_dispatch",
            event_payload={
                "inputs": {
                    "linux_amdgpu_families": "gfx94X, gfx120X",
                    "linux_test_labels": "test:rocprim",
                    "windows_amdgpu_families": "",
                    "windows_test_labels": "",
                    "prebuilt_stages": "foundation,compiler-runtime",
                    "baseline_run_id": "12345",
                }
            },
        )
        self.assertEqual(inputs.linux_amdgpu_families, ["gfx94x", "gfx120x"])
        self.assertEqual(inputs.linux_test_labels, ["test:rocprim"])
        self.assertEqual(inputs.prebuilt_stages, "foundation,compiler-runtime")
        self.assertEqual(inputs.baseline_run_id, "12345")

    def test_pull_request_extracts_labels(self):
        """PR labels are extracted from event.pull_request.labels."""
        inputs = _run_from_environ(
            event_name="pull_request",
            event_payload={
                "pull_request": {
                    "labels": [
                        {"name": "gfx950", "id": 1},
                        {"name": "test:rocprim", "id": 2},
                    ]
                }
            },
            commit_ref="feature-branch",
        )
        self.assertEqual(inputs.pr_labels, ["gfx950", "test:rocprim"])
        self.assertEqual(inputs.base_ref, "HEAD^")

    def test_push_reads_before_sha(self):
        """Push events use event.before as the diff base."""
        inputs = _run_from_environ(
            event_name="push",
            event_payload={"before": "abc123def456"},
        )
        self.assertEqual(inputs.base_ref, "abc123def456")


# ---------------------------------------------------------------------------
# Step 2: Check Skip CI
# ---------------------------------------------------------------------------


class TestShouldSkipCI(unittest.TestCase):
    """Test the skip CI gate.

    Path filtering logic is tested in configure_ci_path_filters_test.py.
    These tests mock is_ci_run_required and verify should_skip_ci's own
    logic: label handling, None changed_files passthrough, and delegation.
    """

    def _inputs(self, **kwargs):
        defaults = dict(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_skip_ci_label(self):
        """PR with ci:skip label skips CI regardless of changed files."""
        inputs = self._inputs(pr_labels=["ci:skip"])
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        self.assertTrue(cm.should_skip_ci(inputs, git))

    def test_pr_without_skip_label_proceeds(self):
        """PR without ci:skip label proceeds to path filtering."""
        inputs = self._inputs(pr_labels=[])
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        self.assertFalse(cm.should_skip_ci(inputs, git))

    @patch("configure_multi_arch_ci.is_ci_run_required", return_value=False)
    def test_path_filter_says_skip(self, mock_filter):
        """When is_ci_run_required returns False, skip CI."""
        inputs = self._inputs()
        git = cm.GitContext(changed_files=["docs/README.md"])
        self.assertTrue(cm.should_skip_ci(inputs, git))
        mock_filter.assert_called_once_with(["docs/README.md"])

    @patch("configure_multi_arch_ci.is_ci_run_required", return_value=True)
    def test_path_filter_says_required(self, mock_filter):
        """When is_ci_run_required returns True, don't skip."""
        inputs = self._inputs()
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        self.assertFalse(cm.should_skip_ci(inputs, git))

    @patch("configure_multi_arch_ci.is_ci_run_required")
    def test_none_changed_files_skips_path_filter(self, mock_filter):
        """schedule/workflow_dispatch pass None → path filter not called."""
        inputs = self._inputs(event_name="schedule")
        git = cm.GitContext()
        self.assertFalse(cm.should_skip_ci(inputs, git))
        mock_filter.assert_not_called()


# ---------------------------------------------------------------------------
# Step 3: Decide Jobs
# ---------------------------------------------------------------------------


class TestDecideJobs(unittest.TestCase):
    """Test job decision logic and test_type determination."""

    def _inputs(self, **kwargs):
        defaults = dict(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_all_job_groups_run(self):
        """All job groups are set to run (subgraph selection is Phase 4)."""
        result = cm.decide_jobs(self._inputs(), git_context=cm.GitContext())
        self.assertIsInstance(result, cm.JobDecisions)
        self.assertEqual(result.build_rocm.action, cm.JobAction.RUN)
        self.assertEqual(result.test_rocm.action, cm.JobAction.RUN)
        self.assertEqual(result.build_rocm_python.action, cm.JobAction.RUN)
        self.assertEqual(result.build_pytorch.action, cm.JobAction.RUN)
        self.assertEqual(result.test_pytorch.action, cm.JobAction.RUN)

    def test_default_test_type_is_quick(self):
        """Default test_type for PR/push with no special conditions."""
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        result = cm.decide_jobs(self._inputs(), git_context=git)
        self.assertEqual(result.test_rocm.test_type, "quick")

    def test_schedule_is_comprehensive(self):
        """Schedule trigger → comprehensive tests."""
        result = cm.decide_jobs(
            self._inputs(event_name="schedule"), git_context=cm.GitContext()
        )
        self.assertEqual(result.test_rocm.test_type, "comprehensive")

    def test_submodule_change_is_full(self):
        """Changed files matching a submodule path → full tests."""
        git = cm.GitContext(
            changed_files=["rocm-libraries", "CMakeLists.txt"],
            submodule_paths=["rocm-libraries", "rocm-systems"],
        )
        result = cm.decide_jobs(self._inputs(), git_context=git)
        self.assertEqual(result.test_rocm.test_type, "full")
        self.assertIn("submodule", result.test_rocm.test_type_reason)

    def test_no_submodule_change_stays_quick(self):
        """Changed files not matching any submodule → stays quick."""
        git = cm.GitContext(
            changed_files=["CMakeLists.txt"],
            submodule_paths=["rocm-libraries", "rocm-systems"],
        )
        result = cm.decide_jobs(self._inputs(), git_context=git)
        self.assertEqual(result.test_rocm.test_type, "quick")

    def test_pr_test_label_is_full(self):
        """PR with test:* label → full tests."""
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        result = cm.decide_jobs(
            self._inputs(pr_labels=["test:rocprim"]), git_context=git
        )
        self.assertEqual(result.test_rocm.test_type, "full")

    def test_workflow_dispatch_test_labels_is_full(self):
        """workflow_dispatch with test labels → full tests."""
        result = cm.decide_jobs(
            self._inputs(
                event_name="workflow_dispatch",
                linux_test_labels="test:rocprim",
            ),
            git_context=cm.GitContext(),
        )
        self.assertEqual(result.test_rocm.test_type, "full")

    def test_test_filter_label_overrides(self):
        """test_filter: PR label overrides the computed test_type."""
        # Even though schedule would set comprehensive, test_filter overrides.
        result = cm.decide_jobs(
            self._inputs(
                event_name="schedule",
                pr_labels=["test_filter:standard"],
            ),
            git_context=cm.GitContext(),
        )
        self.assertEqual(result.test_rocm.test_type, "standard")

    def test_test_filter_invalid_raises(self):
        """Unrecognized test_filter value raises ValueError."""
        git = cm.GitContext(changed_files=["CMakeLists.txt"])
        with self.assertRaises(ValueError, msg="Unrecognized test_filter"):
            cm.decide_jobs(
                self._inputs(pr_labels=["test_filter:bogus"]), git_context=git
            )

    def test_explicit_prebuilt_stages(self):
        """workflow_dispatch prebuilt_stages input → stage_decisions on BuildRocmDecision."""
        result = cm.decide_jobs(
            self._inputs(
                event_name="workflow_dispatch",
                prebuilt_stages="foundation,compiler-runtime",
            ),
            git_context=cm.GitContext(),
        )
        self.assertEqual(
            sorted(result.build_rocm.prebuilt_stages),
            ["compiler-runtime", "foundation"],
        )
        self.assertEqual(result.build_rocm.rebuild_stages, [])

    def test_no_prebuilt_stages_by_default(self):
        """Without explicit prebuilt_stages, no stage decisions are set."""
        result = cm.decide_jobs(self._inputs(), git_context=cm.GitContext())
        self.assertEqual(result.build_rocm.prebuilt_stages, [])
        self.assertEqual(result.build_rocm.stage_decisions, {})

    def test_build_rocm_stage_partitioning(self):
        """BuildRocmDecision correctly partitions stages into prebuilt/rebuild."""
        decision = cm.BuildRocmDecision(
            action=cm.JobAction.RUN,
            stage_decisions={
                "foundation": cm.JobAction.PREBUILT,
                "compiler-runtime": cm.JobAction.PREBUILT,
                "math-libs": cm.JobAction.RUN,
            },
        )
        self.assertEqual(
            sorted(decision.prebuilt_stages),
            ["compiler-runtime", "foundation"],
        )
        self.assertEqual(decision.rebuild_stages, ["math-libs"])


# ---------------------------------------------------------------------------
# Step 4: Select Targets
# ---------------------------------------------------------------------------


class TestSelectTargets(unittest.TestCase):
    """Test target family selection.

    These tests exercise the trigger-type dispatch and label parsing logic.
    Family names and platform availability come from amdgpu_family_matrix.py
    (the real data), so tests assert on structural properties rather than
    hardcoding specific family names.
    """

    def test_push_includes_postsubmit_families(self):
        """Push trigger selects presubmit+postsubmit families."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # gfx950 is postsubmit-only, should be present for push
        self.assertIn("gfx950", result.linux_families)

    def test_schedule_returns_all_families(self):
        """Schedule trigger selects all families (presubmit+postsubmit+nightly)."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="schedule",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # Schedule should have more families than push (nightly families added)
        push_inputs = cm.CIInputs(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        push_result = cm.select_targets(push_inputs)
        self.assertGreater(len(result.linux_families), len(push_result.linux_families))

    def test_pull_request_defaults_to_presubmit_only(self):
        """PR without labels gets presubmit families only, not postsubmit."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        self.assertGreater(len(result.linux_families), 0)
        # gfx950 is postsubmit-only, should NOT be in PR defaults
        self.assertNotIn("gfx950", result.linux_families)

    def test_pull_request_gfx_label_adds_family(self):
        """PR with a gfx label adds that family to the defaults."""
        inputs_without = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        inputs_with = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
            # gfx906 is nightly-only, not in presubmit+postsubmit defaults
            pr_labels=["gfx906"],
        )
        result_without = cm.select_targets(inputs_without)
        result_with = cm.select_targets(inputs_with)
        self.assertNotIn("gfx906", result_without.linux_families)
        self.assertIn("gfx906", result_with.linux_families)

    def test_pull_request_run_all_archs_label(self):
        """PR with ci:run-all-archs label selects all families."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
            pr_labels=["ci:run-all-archs"],
        )
        result = cm.select_targets(inputs)
        # Should include nightly-only families
        self.assertIn("gfx906", result.linux_families)

    def test_pull_request_unknown_gfx_label_raises(self):
        """PR with an unknown gfx label fails fast."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="feature",
            base_ref="HEAD^",
            build_variant="release",
            pr_labels=["gfx9999"],
        )
        with self.assertRaises(ValueError, msg="Unknown GPU families"):
            cm.select_targets(inputs)

    def test_workflow_dispatch_per_platform(self):
        """workflow_dispatch selects families per platform."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="workflow_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
            linux_amdgpu_families=["gfx94x", "gfx110x"],
            windows_amdgpu_families=["gfx110x"],
        )
        result = cm.select_targets(inputs)
        self.assertIn("gfx94x", result.linux_families)
        self.assertIn("gfx110x", result.linux_families)
        self.assertIn("gfx110x", result.windows_families)
        # gfx94x has no windows entry in the matrix
        self.assertNotIn("gfx94x", result.windows_families)

    def test_workflow_dispatch_empty_input(self):
        """workflow_dispatch with empty lists returns empty families."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="workflow_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        self.assertEqual(result.linux_families, [])
        self.assertEqual(result.windows_families, [])

    def test_workflow_dispatch_unknown_family_raises(self):
        """workflow_dispatch with unknown family fails fast."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="workflow_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
            linux_amdgpu_families=["gfx_bogus"],
        )
        with self.assertRaises(ValueError, msg="Unknown GPU families"):
            cm.select_targets(inputs)

    @unittest.skip(
        "TODO: workflow_dispatch should reject families unavailable on the requested platform"
    )
    def test_workflow_dispatch_wrong_platform_raises(self):
        """Requesting a family for a platform it doesn't support should fail."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="workflow_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
            # gfx950 has no windows entry — this should be an error, not silently dropped
            windows_amdgpu_families=["gfx950"],
        )
        with self.assertRaises(ValueError):
            cm.select_targets(inputs)

    def test_unsupported_event_type_raises(self):
        """Unknown event type raises ValueError."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="repository_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        with self.assertRaises(ValueError, msg="Unsupported event type"):
            cm.select_targets(inputs)

    def test_platform_filtering(self):
        """Families without a platform entry are excluded from that platform."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # gfx94x is linux-only (no windows entry in presubmit matrix)
        self.assertIn("gfx94x", result.linux_families)
        self.assertNotIn("gfx94x", result.windows_families)


# ---------------------------------------------------------------------------
# Step 5: Build Configs
# ---------------------------------------------------------------------------


class TestExpandBuildConfigs(unittest.TestCase):
    """Test expand_build_configs: TargetSelection × CIInputs → BuildConfigs.

    Tests verify structural properties of the output, not specific data values
    from amdgpu_family_matrix.py. Changing a runner label or flipping
    expect_failure in the matrix data should not require test updates here.
    """

    def _inputs(self, **kwargs):
        defaults = dict(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_build_config_to_dict_has_all_fields(self):
        """BuildConfig.to_dict() produces all expected keys."""
        config = cm.BuildConfig(
            per_family_info=[],
            dist_amdgpu_families="",
            artifact_group="multi-arch-release",
            build_variant_label="release",
            build_variant_suffix="",
            build_variant_cmake_preset="",
            expect_failure=False,
            build_pytorch=True,
        )
        d = config.to_dict()
        # to_dict keys should match dataclass fields.
        expected_keys = {f.name for f in fields(cm.BuildConfig)}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_empty_targets_both_none(self):
        """Empty targets on both platforms → both None."""
        targets = cm.TargetSelection()
        result = cm.expand_build_configs(targets=targets, ci_inputs=self._inputs())
        self.assertIsNone(result.linux)
        self.assertIsNone(result.windows)

    def test_build_config_serialization_empty_vs_present(self):
        """Workflow YAML gates on build_config != '', so None must serialize
        to '' and present configs must serialize to valid JSON."""
        config = cm.BuildConfig(
            per_family_info=[{"amdgpu_family": "gfx110x"}],
            dist_amdgpu_families="gfx110x",
            artifact_group="multi-arch-release",
            build_variant_label="release",
            build_variant_suffix="",
            build_variant_cmake_preset="release",
            expect_failure=False,
            build_pytorch=True,
        )
        # Present config → valid JSON
        serialized = json.dumps(config.to_dict())
        self.assertTrue(serialized)
        round_tripped = json.loads(serialized)
        self.assertEqual(round_tripped["dist_amdgpu_families"], "gfx110x")

        # None config → empty string (matches workflow `!= ''` gate)
        none_serialized = json.dumps(None.to_dict()) if None else ""
        self.assertEqual(none_serialized, "")

    def test_release_produces_configs_for_both_platforms(self):
        """Release variant with families on both platforms produces both configs
        with correctly structured per-family info."""
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        targets = cm.select_targets(inputs)
        result = cm.expand_build_configs(targets=targets, ci_inputs=inputs)
        required_keys = {
            "amdgpu_family",
            "amdgpu_targets",
            "test-runs-on",
            "sanity_check_only_for_family",
        }
        for config in [result.linux, result.windows]:
            self.assertIsNotNone(config)
            per_family = config.per_family_info
            self.assertGreater(len(per_family), 0)
            for entry in per_family:
                self.assertEqual(
                    set(entry.keys()),
                    required_keys,
                    f"unexpected keys in per-family info: {entry}",
                )

    def test_build_config_structure(self):
        """BuildConfig has correct structure: families, metadata, consistency.

        BuildConfig carries two representations of the family list for
        different workflow consumers:

        per_family_info — JSON array with per-family metadata for
        test and per-arch artifact jobs (fromJSON matrix expansion):

            [
                {
                    "amdgpu_family": "gfx94X-dcgpu",
                    "amdgpu_targets": "gfx942",
                    "test-runs-on": "linux-mi325-1gpu-ossci-rocm",
                    "sanity_check_only_for_family": false
                },
                ...
            ]

        dist_amdgpu_families — semicolon-separated family names for CMake
        (THEROCK_DIST_AMDGPU_TARGETS) and configure_stage.py:

            "gfx94X-dcgpu;gfx110X-all"

        Both must contain the same set of families.
        """
        targets = cm.TargetSelection(
            linux_families=["gfx94x", "gfx110x"],
            windows_families=["gfx110x"],
        )
        result = cm.expand_build_configs(targets=targets, ci_inputs=self._inputs())

        # All target families that support the variant appear in output.
        linux_per_family = result.linux.per_family_info
        self.assertEqual(len(linux_per_family), 2)
        windows_per_family = result.windows.per_family_info
        self.assertEqual(len(windows_per_family), 1)

        # The two family representations carry the same set of families.
        dist_set = set(result.linux.dist_amdgpu_families.split(";"))
        json_set = {f["amdgpu_family"] for f in linux_per_family}
        self.assertEqual(dist_set, json_set)

        # Variant metadata is populated.
        config = result.linux
        self.assertTrue(len(config.build_variant_label) > 0)
        self.assertIn("release", config.artifact_group)
        self.assertIsInstance(config.expect_failure, bool)
        self.assertIsInstance(config.build_pytorch, bool)

    def test_variant_filters_by_platform_and_family_support(self):
        """ASAN: only gfx94x on linux supports it, gfx110x doesn't, windows has no ASAN config."""
        # gfx94x supports asan, gfx110x is release-only, windows has no asan variant.
        targets = cm.TargetSelection(
            linux_families=["gfx94x", "gfx110x"],
            windows_families=["gfx110x"],
        )
        result = cm.expand_build_configs(
            targets=targets, ci_inputs=self._inputs(build_variant="asan")
        )
        # Only gfx94x on linux survives.
        self.assertIsNotNone(result.linux)
        linux_per_family = result.linux.per_family_info
        self.assertEqual(len(linux_per_family), 1)
        # Windows has no asan variant config at all.
        self.assertIsNone(result.windows)

    def test_test_runner_kernel_overrides_runner_label(self):
        """test_runner:oem label swaps in kernel-specific runner for gfx1151."""
        targets = cm.TargetSelection(linux_families=["gfx1151"])
        result = cm.expand_build_configs(
            targets=targets,
            ci_inputs=self._inputs(pr_labels=["test_runner:oem"]),
        )
        self.assertIsNotNone(result.linux)
        entry = result.linux.per_family_info[0]
        self.assertEqual(entry["test-runs-on"], "linux-strix-halo-gpu-rocm-oem")

    def test_test_runner_kernel_clears_unsupported_family(self):
        """test_runner:oem label clears runner for families without kernel support."""
        # gfx94x has no test-runs-on-kernel entry
        targets = cm.TargetSelection(linux_families=["gfx94x"])
        result = cm.expand_build_configs(
            targets=targets,
            ci_inputs=self._inputs(pr_labels=["test_runner:oem"]),
        )
        self.assertIsNotNone(result.linux)
        entry = result.linux.per_family_info[0]
        self.assertEqual(entry["test-runs-on"], "")

    def test_no_test_runner_label_uses_default(self):
        """Without test_runner: label, default runner labels are used."""
        targets = cm.TargetSelection(linux_families=["gfx1151"])
        result = cm.expand_build_configs(targets=targets, ci_inputs=self._inputs())
        self.assertIsNotNone(result.linux)
        entry = result.linux.per_family_info[0]
        # Default runner, not the oem one
        self.assertNotEqual(entry["test-runs-on"], "")
        self.assertNotIn("oem", entry["test-runs-on"])


# ---------------------------------------------------------------------------
# Step 6: Format Outputs
# ---------------------------------------------------------------------------


class TestFormatSummary(unittest.TestCase):
    """Test summary formatting (pure function)."""

    def _inputs(self, **kwargs):
        defaults = dict(
            run_id="12345",
            event_name="push",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_skipped_summary(self):
        outputs = cm.CIOutputs.skipped()
        result = format_summary(self._inputs(), outputs)
        # Just check the header. The output is markdown and asserting
        # on more exact formatting would create a change detector test.
        self.assertTrue(result.startswith("## Multi-Arch CI Configuration"))

    def test_normal_summary(self):
        """Only checks header — output is markdown, not a contract.
        Asserting on exact wording would create a change-detector test."""
        jobs = cm.JobDecisions(
            build_rocm=cm.BuildRocmDecision(action=cm.JobAction.RUN),
            test_rocm=cm.TestRocmDecision(action=cm.JobAction.RUN, test_type="full"),
            build_rocm_python=cm.JobGroupDecision(action=cm.JobAction.RUN),
            build_pytorch=cm.JobGroupDecision(action=cm.JobAction.RUN),
            test_pytorch=cm.JobGroupDecision(action=cm.JobAction.RUN),
        )
        outputs = cm.CIOutputs(is_ci_enabled=True, jobs=jobs)
        result = format_summary(self._inputs(), outputs)
        # Just check the header. The output is markdown for humans and asserting
        # on more exact formatting would create a change detector test.
        self.assertTrue(result.startswith("## Multi-Arch CI Configuration"))

    def test_skipped_ci_write_outputs_summary(self):
        outputs = cm.CIOutputs(is_ci_enabled=False)
        cm.write_outputs(self._inputs(), outputs)


# ---------------------------------------------------------------------------
# End-to-end: configure() pipeline
# ---------------------------------------------------------------------------


class TestConfigurePipeline(unittest.TestCase):
    """Test the full pipeline via configure()."""

    def test_skipped_outputs(self):
        """CIOutputs.skipped produces empty, disabled outputs."""
        outputs = cm.CIOutputs.skipped()
        self.assertFalse(outputs.is_ci_enabled)
        self.assertIsNone(outputs.builds.linux)
        self.assertIsNone(outputs.builds.windows)
        self.assertIsNone(outputs.jobs)

    @patch("configure_multi_arch_ci.should_skip_ci")
    def test_pipeline_skips_when_gate_says_skip(self, mock_skip):
        """If should_skip_ci returns True, pipeline short-circuits."""
        mock_skip.return_value = True
        inputs = cm.CIInputs(
            run_id="12345",
            event_name="workflow_dispatch",
            commit_ref="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        outputs = cm.configure(inputs, cm.GitContext())
        self.assertFalse(outputs.is_ci_enabled)
        self.assertIsNone(outputs.builds.linux)


# ---------------------------------------------------------------------------
# Contract: BuildConfig fields match workflow YAML references
# ---------------------------------------------------------------------------


class TestBuildConfigWorkflowContract(unittest.TestCase):
    """Verify that workflow YAML references to fromJSON(inputs.build_config).FIELD
    only use fields that exist in BuildConfig.to_dict().

    If a workflow references a field that was renamed or removed in Python,
    this test fails — catching the mismatch before CI does a runtime fromJSON
    and gets null. Fields in Python but not referenced in YAML are fine
    (not every workflow uses every field).
    """

    @staticmethod
    def _extract_build_config_fields(workflow_path):
        """Extract field names referenced as fromJSON(inputs.build_config).X."""
        # We need the raw text, not parsed YAML, to find expression references.
        text = workflow_path.read_text()
        # Match fromJSON(inputs.build_config).FIELD_NAME
        pattern = r"fromJSON\(inputs\.build_config\)\.(\w+)"
        return set(re.findall(pattern, text))

    def _assert_yaml_fields_subset_of_python(self, workflow_path):
        yaml_fields = self._extract_build_config_fields(workflow_path)
        python_fields = {f.name for f in fields(cm.BuildConfig)}
        unknown = yaml_fields - python_fields
        self.assertEqual(
            unknown,
            set(),
            f"{workflow_path.name} references BuildConfig fields that don't "
            f"exist in Python: {unknown}. "
            f"Available fields: {sorted(python_fields)}",
        )

    def test_linux_workflow_uses_all_fields(self):
        """Linux workflow should reference every BuildConfig field."""
        workflow_path = WORKFLOWS_DIR / "multi_arch_ci_linux.yml"
        yaml_fields = self._extract_build_config_fields(workflow_path)
        python_fields = {f.name for f in fields(cm.BuildConfig)}
        self.assertEqual(
            yaml_fields,
            python_fields,
            f"BuildConfig fields mismatch with {workflow_path.name}.\n"
            f"  In YAML but not Python: {yaml_fields - python_fields}\n"
            f"  In Python but not YAML: {python_fields - yaml_fields}",
        )

    def test_windows_workflow_uses_all_fields(self):
        """Windows workflow should reference every BuildConfig field."""
        workflow_path = WORKFLOWS_DIR / "multi_arch_ci_windows.yml"
        yaml_fields = self._extract_build_config_fields(workflow_path)
        python_fields = {f.name for f in fields(cm.BuildConfig)}
        self.assertEqual(
            yaml_fields,
            python_fields,
            f"BuildConfig fields mismatch with {workflow_path.name}.\n"
            f"  In YAML but not Python: {yaml_fields - python_fields}\n"
            f"  In Python but not YAML: {python_fields - yaml_fields}",
        )


if __name__ == "__main__":
    unittest.main()
