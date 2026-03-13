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
# CIInputs — construction and properties
# ---------------------------------------------------------------------------


class TestCIInputs(unittest.TestCase):
    """Test CIInputs dataclass and its properties."""

    def test_pull_request_properties(self):
        inputs = cm.CIInputs(
            event_name="pull_request",
            branch_name="feature-branch",
            base_ref="HEAD^",
            build_variant="release",
            pr_labels=["gfx950", "test:rocprim"],
        )
        self.assertTrue(inputs.is_pull_request)
        self.assertFalse(inputs.is_push)
        self.assertFalse(inputs.is_schedule)
        self.assertFalse(inputs.is_workflow_dispatch)

    def test_push_properties(self):
        inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="abc123",
            build_variant="release",
        )
        self.assertFalse(inputs.is_pull_request)
        self.assertTrue(inputs.is_push)

    def test_schedule_properties(self):
        inputs = cm.CIInputs(
            event_name="schedule",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        self.assertTrue(inputs.is_schedule)

    def test_workflow_dispatch_properties(self):
        inputs = cm.CIInputs(
            event_name="workflow_dispatch",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
            linux_amdgpu_families="gfx94X, gfx120X",
        )
        self.assertTrue(inputs.is_workflow_dispatch)
        self.assertEqual(inputs.linux_amdgpu_families, "gfx94X, gfx120X")

    def test_defaults(self):
        """Fields with defaults can be omitted."""
        inputs = cm.CIInputs(
            event_name="push",
            branch_name="main",
            base_ref="HEAD^1",
            build_variant="release",
        )
        self.assertEqual(inputs.pr_labels, [])
        self.assertEqual(inputs.linux_amdgpu_families, "")
        self.assertEqual(inputs.prebuilt_stages, "")


class TestCIInputsFromEnviron(unittest.TestCase):
    """Test CIInputs.from_environ() with event payload fixtures."""

    def test_workflow_dispatch_event(self):
        """from_environ reads workflow_dispatch inputs from GITHUB_EVENT_PATH."""
        event_payload = {
            "inputs": {
                "linux_amdgpu_families": "gfx94X, gfx120X",
                "linux_test_labels": "test:rocprim",
                "windows_amdgpu_families": "",
                "windows_test_labels": "",
                "prebuilt_stages": "foundation,compiler-runtime",
                "baseline_run_id": "12345",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event_payload, f)
            event_path = f.name

        try:
            env = {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_EVENT_PATH": event_path,
                "GITHUB_REF_NAME": "main",
                "BUILD_VARIANT": "release",
            }
            with patch.dict(os.environ, env, clear=False):
                inputs = cm.CIInputs.from_environ()

            self.assertEqual(inputs.event_name, "workflow_dispatch")
            self.assertEqual(inputs.linux_amdgpu_families, "gfx94X, gfx120X")
            self.assertEqual(inputs.linux_test_labels, "test:rocprim")
            self.assertEqual(inputs.prebuilt_stages, "foundation,compiler-runtime")
            self.assertEqual(inputs.baseline_run_id, "12345")
        finally:
            os.unlink(event_path)

    def test_pull_request_event_with_labels(self):
        """from_environ extracts PR labels from the event payload."""
        event_payload = {
            "pull_request": {
                "labels": [
                    {"name": "gfx950", "id": 1},
                    {"name": "test:rocprim", "id": 2},
                ]
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event_payload, f)
            event_path = f.name

        try:
            env = {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_EVENT_PATH": event_path,
                "GITHUB_REF_NAME": "feature-branch",
                "BUILD_VARIANT": "release",
            }
            with patch.dict(os.environ, env, clear=False):
                inputs = cm.CIInputs.from_environ()

            self.assertEqual(inputs.event_name, "pull_request")
            self.assertEqual(inputs.pr_labels, ["gfx950", "test:rocprim"])
            self.assertEqual(inputs.base_ref, "HEAD^")
        finally:
            os.unlink(event_path)

    def test_push_event(self):
        """from_environ reads 'before' SHA for push events."""
        event_payload = {"before": "abc123def456"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event_payload, f)
            event_path = f.name

        try:
            env = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_EVENT_PATH": event_path,
                "GITHUB_REF_NAME": "main",
                "BUILD_VARIANT": "release",
            }
            with patch.dict(os.environ, env, clear=False):
                inputs = cm.CIInputs.from_environ()

            self.assertEqual(inputs.base_ref, "abc123def456")
        finally:
            os.unlink(event_path)


# ---------------------------------------------------------------------------
# Step 2: Check Skip CI
# ---------------------------------------------------------------------------


class TestCheckSkipCI(unittest.TestCase):
    """Test the skip CI gate."""

    def _make_inputs(self, **kwargs) -> cm.CIInputs:
        defaults = {
            "event_name": "pull_request",
            "branch_name": "feature",
            "base_ref": "HEAD^",
            "build_variant": "release",
        }
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_no_skip_by_default(self):
        """Default stub does not skip."""
        inputs = self._make_inputs()
        result = cm.check_skip_ci(inputs, changed_files=["some/file.cpp"])
        self.assertFalse(result.skip)

    # TODO: Tests for skip-ci label, docs-only changes, no files changed
    # These will be filled in when check_skip_ci is implemented (Phase 2).


# ---------------------------------------------------------------------------
# Step 3: Select Targets
# ---------------------------------------------------------------------------


class TestSelectTargets(unittest.TestCase):
    """Test target family selection."""

    def _make_inputs(self, **kwargs) -> cm.CIInputs:
        defaults = {
            "event_name": "push",
            "branch_name": "main",
            "base_ref": "HEAD^1",
            "build_variant": "release",
        }
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_returns_target_selection(self):
        """Stub returns a TargetSelection dataclass."""
        inputs = self._make_inputs()
        result = cm.select_targets(inputs)
        self.assertIsInstance(result, cm.TargetSelection)

    # TODO: Tests for each trigger type, label parsing, family validation
    # These will be filled in when select_targets is implemented (Phase 2).


# ---------------------------------------------------------------------------
# Step 4: Decide Jobs
# ---------------------------------------------------------------------------


class TestDecideJobs(unittest.TestCase):
    """Test job decision logic."""

    def _make_inputs(self, **kwargs) -> cm.CIInputs:
        defaults = {
            "event_name": "push",
            "branch_name": "main",
            "base_ref": "HEAD^1",
            "build_variant": "release",
        }
        defaults.update(kwargs)
        return cm.CIInputs(**defaults)

    def test_stub_returns_job_decisions(self):
        """Stub returns JobDecisions with all groups set to run."""
        inputs = self._make_inputs()
        result = cm.decide_jobs(inputs, changed_files=None)
        self.assertIsInstance(result, cm.JobDecisions)
        self.assertEqual(result.build_rocm.action, "run")
        self.assertEqual(result.test_rocm.action, "run")
        self.assertEqual(result.build_rocm_python.action, "run")
        self.assertEqual(result.build_pytorch.action, "run")
        self.assertEqual(result.test_pytorch.action, "run")

    def test_test_rocm_has_test_type(self):
        """TestRocmDecision carries test_type details."""
        inputs = self._make_inputs()
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
# Step 5: Expand Matrix
# ---------------------------------------------------------------------------


class TestExpandMatrix(unittest.TestCase):
    """Test matrix expansion."""

    def test_empty_families_returns_empty(self):
        """No families → no matrix entries."""
        result = cm.expand_matrix([], "linux", "release")
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

    # TODO: Tests for actual family expansion logic (Phase 2).


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
    @patch("configure_multi_arch_ci.expand_matrix")
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
        mock_expand.return_value = []

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
        # expand_matrix called twice: once for linux, once for windows
        self.assertEqual(mock_expand.call_count, 2)


if __name__ == "__main__":
    unittest.main()
