# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_multi_arch_ci.py.

Each test demonstrates the pattern for testing a pipeline step:
construct the input dataclass, call the function, assert on the output.
No environment variables or filesystem access needed (except from_environ tests).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_multi_arch_ci as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_from_environ(
    event_name: str,
    event_payload: dict,
    *,
    branch_name: str = "main",
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
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_REF_NAME": branch_name,
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
            event_name="pull_request",
            branch_name="feature",
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
            event_name="push",
            branch_name="main",
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
        self.assertEqual(inputs.linux_amdgpu_families, ["gfx94X", "gfx120X"])
        self.assertEqual(inputs.linux_test_labels, "test:rocprim")
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
            branch_name="feature-branch",
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


class TestCheckSkipCI(unittest.TestCase):
    """Test the skip CI gate."""

    def test_no_skip_by_default(self):
        """Default stub does not skip."""
        inputs = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        result = cm.check_skip_ci(inputs, changed_files=["some/file.cpp"])
        self.assertFalse(result.skip)

    # TODO: Tests for skip-ci label, docs-only changes, no files changed
    # These will be filled in when check_skip_ci is implemented (Phase 2).


# ---------------------------------------------------------------------------
# Step 3: Decide Jobs
# ---------------------------------------------------------------------------


class TestDecideJobs(unittest.TestCase):
    """Test job decision logic."""

    def test_stub_returns_job_decisions(self):
        """Stub returns JobDecisions with all groups set to run."""
        inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.decide_jobs(inputs, changed_files=None)
        self.assertIsInstance(result, cm.JobDecisions)
        self.assertEqual(result.build_rocm.action, "run")
        self.assertEqual(result.test_rocm.action, "run")
        self.assertEqual(result.build_rocm_python.action, "run")
        self.assertEqual(result.build_pytorch.action, "run")
        self.assertEqual(result.test_pytorch.action, "run")

    def test_test_rocm_has_test_type(self):
        """TestRocmDecision carries test_type details."""
        inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.decide_jobs(inputs, changed_files=None)
        self.assertIsInstance(result.test_rocm, cm.TestRocmDecision)
        self.assertEqual(result.test_rocm.test_type, "smoke")

    def test_build_rocm_stage_partitioning(self):
        """BuildRocmDecision correctly partitions stages into prebuilt/rebuild."""
        decision = cm.BuildRocmDecision(
            action="run",
            reason="source changes",
            stage_decisions={
                "foundation": cm.StageDecision(action="prebuilt", reason="no changes"),
                "compiler-runtime": cm.StageDecision(
                    action="prebuilt", reason="no changes"
                ),
                "math-libs": cm.StageDecision(
                    action="rebuild", reason="rocm-libraries changed"
                ),
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
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # gfx950 is postsubmit-only, should be present for push
        self.assertIn("gfx950", result.linux_families)

    def test_schedule_returns_all_families(self):
        """Schedule trigger selects all families (presubmit+postsubmit+nightly)."""
        inputs = cm.CIInputs(
            event_name="schedule",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # Schedule should have more families than push (nightly families added)
        push_inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        push_result = cm.select_targets(push_inputs)
        self.assertGreater(len(result.linux_families), len(push_result.linux_families))

    def test_pull_request_defaults_to_presubmit_only(self):
        """PR without labels gets presubmit families only, not postsubmit."""
        inputs = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature",
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
            event_name="pull_request",
            branch_name="feature",
            base_ref="HEAD^",
            build_variant="release",
        )
        inputs_with = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature",
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
        """PR with run-all-archs-ci label selects all families."""
        inputs = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature",
            base_ref="HEAD^",
            build_variant="release",
            pr_labels=["run-all-archs-ci"],
        )
        result = cm.select_targets(inputs)
        # Should include nightly-only families
        self.assertIn("gfx906", result.linux_families)

    def test_pull_request_unknown_gfx_label_raises(self):
        """PR with an unknown gfx label fails fast."""
        inputs = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature",
            base_ref="HEAD^",
            build_variant="release",
            pr_labels=["gfx9999"],
        )
        with self.assertRaises(ValueError, msg="Unknown GPU families"):
            cm.select_targets(inputs)

    def test_workflow_dispatch_per_platform(self):
        """workflow_dispatch selects families per platform."""
        inputs = cm.CIInputs(
            event_name="workflow_dispatch",
            branch_name="main",
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
            event_name="workflow_dispatch",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        self.assertEqual(result.linux_families, [])
        self.assertEqual(result.windows_families, [])

    def test_workflow_dispatch_unknown_family_raises(self):
        """workflow_dispatch with unknown family fails fast."""
        inputs = cm.CIInputs(
            event_name="workflow_dispatch",
            branch_name="main",
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
            event_name="workflow_dispatch",
            branch_name="main",
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
            event_name="repository_dispatch",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        with self.assertRaises(ValueError, msg="Unsupported event type"):
            cm.select_targets(inputs)

    def test_platform_filtering(self):
        """Families without a platform entry are excluded from that platform."""
        inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        result = cm.select_targets(inputs)
        # gfx94x is linux-only (no windows entry in presubmit matrix)
        self.assertIn("gfx94x", result.linux_families)
        self.assertNotIn("gfx94x", result.windows_families)


# ---------------------------------------------------------------------------
# Step 5: Expand Matrix
# ---------------------------------------------------------------------------


class TestExpandMatrix(unittest.TestCase):
    """Test matrix expansion (_expand_matrix_for_platform and expand_matrices)."""

    def _expand(self, families, platform, build_variant):
        """Helper: look up variant config and call _expand_matrix_for_platform."""
        from amdgpu_family_matrix import (
            all_build_variants,
            get_all_families_for_trigger_types,
        )

        all_families = get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
        variant_config = all_build_variants.get(platform, {}).get(build_variant)
        if not variant_config:
            return []
        return cm._expand_matrix_for_platform(
            families=families,
            platform=platform,
            build_variant=build_variant,
            all_families=all_families,
            variant_config=variant_config,
        )

    def test_empty_families_returns_empty(self):
        """No families → no matrix entries."""
        result = self._expand([], "linux", "release")
        self.assertEqual(result, [])

    def test_matrix_entry_to_dict(self):
        """MatrixEntry.to_dict() produces the expected structure."""
        entry = cm.MatrixEntry(
            matrix_per_family_json='[{"amdgpu_family": "gfx94X-dcgpu"}]',
            dist_amdgpu_families="gfx94X-dcgpu",
            artifact_group="multi-arch-release",
            build_variant_label="release",
            build_variant_suffix="",
            build_variant_cmake_preset="",
            expect_failure=False,
            build_pytorch=True,
        )
        d = entry.to_dict()
        self.assertEqual(d["artifact_group"], "multi-arch-release")
        self.assertFalse(d["expect_failure"])
        self.assertTrue(d["build_pytorch"])

    def test_linux_release_presubmit_families(self):
        """Presubmit families on linux/release produces one entry with all families."""
        result = self._expand(
            ["gfx94x", "gfx110x", "gfx1151", "gfx120x"], "linux", "release"
        )
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry.build_variant_label, "release")
        self.assertEqual(entry.build_variant_suffix, "")
        self.assertEqual(entry.artifact_group, "multi-arch-release")
        self.assertFalse(entry.expect_failure)
        self.assertTrue(entry.build_pytorch)

        # All four presubmit families should appear in the per-family JSON.
        per_family = json.loads(entry.matrix_per_family_json)
        family_names = [f["amdgpu_family"] for f in per_family]
        self.assertIn("gfx94X-dcgpu", family_names)
        self.assertIn("gfx110X-all", family_names)
        self.assertIn("gfx1151", family_names)
        self.assertIn("gfx120X-all", family_names)

        # dist_amdgpu_families is semicolon-separated family names.
        self.assertEqual(
            entry.dist_amdgpu_families,
            ";".join(family_names),
        )

    def test_windows_release_presubmit_families(self):
        """Windows release includes only families with a windows platform entry."""
        result = self._expand(
            ["gfx94x", "gfx110x", "gfx1151", "gfx120x"], "windows", "release"
        )
        self.assertEqual(len(result), 1)
        per_family = json.loads(result[0].matrix_per_family_json)
        family_names = [f["amdgpu_family"] for f in per_family]
        # gfx94x has no windows entry in the matrix.
        self.assertNotIn("gfx94X-dcgpu", family_names)
        # gfx110x, gfx1151, gfx120x have windows entries.
        self.assertIn("gfx110X-all", family_names)
        self.assertIn("gfx1151", family_names)
        self.assertIn("gfx120X-all", family_names)

    def test_per_family_info_fields(self):
        """Per-family info contains expected fields."""
        result = self._expand(["gfx94x"], "linux", "release")
        self.assertEqual(len(result), 1)
        per_family = json.loads(result[0].matrix_per_family_json)
        self.assertEqual(len(per_family), 1)
        info = per_family[0]
        self.assertEqual(info["amdgpu_family"], "gfx94X-dcgpu")
        self.assertEqual(info["amdgpu_targets"], "gfx942")
        self.assertEqual(info["test-runs-on"], "linux-mi325-1gpu-ossci-rocm")
        self.assertFalse(info["sanity_check_only_for_family"])

    def test_sanity_check_flag_propagated(self):
        """sanity_check_only_for_family flows through to per-family info."""
        result = self._expand(["gfx110x"], "linux", "release")
        per_family = json.loads(result[0].matrix_per_family_json)
        self.assertTrue(per_family[0]["sanity_check_only_for_family"])

    def test_variant_not_supported_by_family(self):
        """A family that doesn't support the requested variant is excluded."""
        # gfx110x only supports "release", not "asan".
        result = self._expand(["gfx110x"], "linux", "asan")
        self.assertEqual(result, [])

    def test_asan_variant(self):
        """ASAN variant produces an entry with correct metadata."""
        result = self._expand(["gfx94x"], "linux", "asan")
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry.build_variant_label, "asan")
        self.assertEqual(entry.build_variant_suffix, "asan")
        self.assertEqual(entry.artifact_group, "multi-arch-asan")
        self.assertEqual(entry.build_variant_cmake_preset, "linux-release-asan")
        self.assertFalse(entry.expect_failure)
        self.assertTrue(entry.build_pytorch)

    def test_tsan_variant_expect_failure(self):
        """TSAN variant has expect_failure=True and build_pytorch=False."""
        result = self._expand(["gfx94x"], "linux", "tsan")
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry.build_variant_label, "tsan")
        self.assertTrue(entry.expect_failure)
        self.assertFalse(entry.build_pytorch)

    def test_unknown_family_skipped(self):
        """A family not in the matrix is silently skipped."""
        result = self._expand(["gfx_nonexistent"], "linux", "release")
        self.assertEqual(result, [])

    def test_unknown_platform_returns_empty(self):
        """A platform with no build variants returns empty."""
        result = self._expand(["gfx94x"], "macos", "release")
        self.assertEqual(result, [])

    def test_nightly_family_expect_pytorch_failure(self):
        """expect_pytorch_failure is per-family data, not per-variant config."""
        # gfx906 on windows has expect_pytorch_failure in the family matrix data,
        # but that's per-family, not per-variant. The variant config for
        # windows/release doesn't set expect_pytorch_failure, so build_pytorch
        # is True at the matrix level.
        result = self._expand(["gfx906"], "windows", "release")
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].build_pytorch)

    def test_multiple_fetch_gfx_targets(self):
        """Multiple fetch-gfx-targets are comma-joined in amdgpu_targets."""
        result = self._expand(["gfx120x"], "linux", "release")
        per_family = json.loads(result[0].matrix_per_family_json)
        # gfx120x has fetch-gfx-targets: ["gfx1200", "gfx1201"]
        self.assertEqual(per_family[0]["amdgpu_targets"], "gfx1200,gfx1201")

    # -- expand_matrices (both-platform wrapper) --

    def test_expand_matrices_both_platforms(self):
        """expand_matrices returns entries for both linux and windows."""
        targets = cm.TargetSelection(
            linux_families=["gfx94x", "gfx110x"],
            windows_families=["gfx110x", "gfx1151"],
        )
        result = cm.expand_matrices(targets=targets, build_variant="release")
        self.assertEqual(len(result.linux_variants), 1)
        self.assertEqual(len(result.windows_variants), 1)

        linux_families = json.loads(result.linux_variants[0].matrix_per_family_json)
        linux_names = [f["amdgpu_family"] for f in linux_families]
        self.assertIn("gfx94X-dcgpu", linux_names)
        self.assertIn("gfx110X-all", linux_names)

        windows_families = json.loads(result.windows_variants[0].matrix_per_family_json)
        windows_names = [f["amdgpu_family"] for f in windows_families]
        self.assertIn("gfx110X-all", windows_names)
        self.assertIn("gfx1151", windows_names)

    def test_expand_matrices_variant_not_on_windows(self):
        """ASAN has no windows config → windows list is empty."""
        targets = cm.TargetSelection(
            linux_families=["gfx94x"],
            windows_families=["gfx110x"],
        )
        result = cm.expand_matrices(targets=targets, build_variant="asan")
        self.assertEqual(len(result.linux_variants), 1)
        self.assertEqual(result.windows_variants, [])

    def test_expand_matrices_empty_targets(self):
        """Empty targets on both platforms → both lists empty."""
        targets = cm.TargetSelection()
        result = cm.expand_matrices(targets=targets, build_variant="release")
        self.assertEqual(result.linux_variants, [])
        self.assertEqual(result.windows_variants, [])

    # -- Parity test --

    def test_output_matches_generate_multi_arch_matrix(self):
        """expand_matrices output matches configure_ci.generate_multi_arch_matrix.

        Parity test: the new functions should produce identical output
        to the old one for the same inputs.
        """
        from amdgpu_family_matrix import (
            all_build_variants,
            get_all_families_for_trigger_types,
        )
        from configure_ci import generate_multi_arch_matrix

        families = ["gfx94x", "gfx110x", "gfx1151", "gfx120x"]
        variant = "release"
        lookup_matrix = get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )

        targets = cm.TargetSelection(
            linux_families=families,
            windows_families=families,
        )
        result = cm.expand_matrices(targets=targets, build_variant=variant)

        for platform, new_result in [
            ("linux", result.linux_variants),
            ("windows", result.windows_variants),
        ]:
            old_result = generate_multi_arch_matrix(
                target_names=families,
                lookup_matrix=lookup_matrix,
                platform=platform,
                platform_build_variants=all_build_variants[platform],
                base_args={"build_variant": variant},
            )
            new_as_dicts = [entry.to_dict() for entry in new_result]
            self.assertEqual(
                len(old_result), len(new_as_dicts), f"length mismatch on {platform}"
            )
            for old_entry, new_entry in zip(old_result, new_as_dicts):
                self.assertEqual(old_entry, new_entry, f"entry mismatch on {platform}")


# ---------------------------------------------------------------------------
# Step 6: Format Outputs
# ---------------------------------------------------------------------------


class TestFormatSummary(unittest.TestCase):
    """Test summary formatting (pure function)."""

    def test_skipped_summary(self):
        outputs = cm.CIOutputs.skipped("only .md files changed")
        summary = cm.format_summary(outputs)
        self.assertIn("is_ci_enabled", summary)
        self.assertIn("False", summary)

    def test_normal_summary(self):
        jobs = cm.JobDecisions(
            build_rocm=cm.BuildRocmDecision(action="run", reason="default"),
            test_rocm=cm.TestRocmDecision(
                action="run", reason="default", test_type="full"
            ),
            build_rocm_python=cm.JobGroupDecision(action="run", reason="default"),
            build_pytorch=cm.JobGroupDecision(action="run", reason="default"),
            test_pytorch=cm.JobGroupDecision(action="run", reason="default"),
        )
        outputs = cm.CIOutputs(is_ci_enabled=True, jobs=jobs)
        summary = cm.format_summary(outputs)
        self.assertIn("True", summary)
        self.assertIn("full", summary)
        self.assertIn("build_rocm", summary)


# ---------------------------------------------------------------------------
# End-to-end: configure() pipeline
# ---------------------------------------------------------------------------


class TestConfigurePipeline(unittest.TestCase):
    """Test the full pipeline via configure()."""

    def test_skipped_outputs(self):
        """CIOutputs.skipped produces empty, disabled outputs."""
        outputs = cm.CIOutputs.skipped("test reason")
        self.assertFalse(outputs.is_ci_enabled)
        self.assertEqual(outputs.linux_variants, [])
        self.assertEqual(outputs.windows_variants, [])

    @patch("configure_multi_arch_ci.check_skip_ci")
    def test_pipeline_skips_when_gate_says_skip(self, mock_skip):
        """If check_skip_ci returns skip=True, pipeline short-circuits."""
        mock_skip.return_value = cm.SkipDecision(skip=True, reason="skip-ci label")
        inputs = cm.CIInputs(
            event_name="workflow_dispatch",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        outputs = cm.configure(inputs)
        self.assertFalse(outputs.is_ci_enabled)
        self.assertEqual(outputs.linux_variants, [])

    @patch("configure_multi_arch_ci.check_skip_ci")
    @patch("configure_multi_arch_ci.select_targets")
    @patch("configure_multi_arch_ci.decide_jobs")
    @patch("configure_multi_arch_ci.expand_matrices")
    def test_pipeline_calls_all_steps(
        self, mock_expand, mock_jobs, mock_targets, mock_skip
    ):
        """When not skipped, all pipeline steps are called."""
        mock_skip.return_value = cm.SkipDecision(skip=False, reason="")
        mock_targets.return_value = cm.TargetSelection(
            linux_families=["gfx94x"],
            windows_families=[],
        )
        mock_jobs.return_value = cm.JobDecisions(
            build_rocm=cm.BuildRocmDecision(action="run", reason="default"),
            test_rocm=cm.TestRocmDecision(action="run", reason="default"),
            build_rocm_python=cm.JobGroupDecision(action="run", reason="default"),
            build_pytorch=cm.JobGroupDecision(action="run", reason="default"),
            test_pytorch=cm.JobGroupDecision(action="run", reason="default"),
        )
        mock_expand.return_value = cm.MatrixExpansion()

        inputs = cm.CIInputs(
            event_name="workflow_dispatch",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        outputs = cm.configure(inputs)

        self.assertTrue(outputs.is_ci_enabled)
        self.assertIsNotNone(outputs.jobs)
        mock_targets.assert_called_once()
        mock_jobs.assert_called_once()
        mock_expand.assert_called_once()


if __name__ == "__main__":
    unittest.main()
