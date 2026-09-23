#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for data invariants in amdgpu_family_matrix.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

# Clear CI_CONFIG_PATH before importing to ensure tests use local definitions only.
# This prevents external config from affecting test results.
if "CI_CONFIG_PATH" in os.environ:
    del os.environ["CI_CONFIG_PATH"]

import amdgpu_family_matrix
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    get_build_runner_labels,
    load_external_runner_config,
    select_build_runner,
)


def _get_all_families_local_only():
    """Get all families using only local definitions (no external config)."""
    # Ensure CI_CONFIG_PATH is not set so we use local definitions
    orig_env = os.environ.get("CI_CONFIG_PATH")
    if "CI_CONFIG_PATH" in os.environ:
        del os.environ["CI_CONFIG_PATH"]
    try:
        return get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
    finally:
        if orig_env is not None:
            os.environ["CI_CONFIG_PATH"] = orig_env


# Load families using local definitions only for invariant tests
ALL_FAMILIES = _get_all_families_local_only()


class TestFamilyMatrixInvariants(unittest.TestCase):
    """Validate structural invariants on the family matrix data."""

    def test_no_duplicate_family_names_per_platform(self):
        """Each (platform, family) pair must be unique across target names.

        Two target names mapping to the same amdgpu_family on the same
        platform would cause silent data loss in matrix expansion.
        """
        for platform in ("linux", "windows"):
            seen: dict[str, str] = {}  # family → target_name
            for target_name, entry in ALL_FAMILIES.items():
                if platform not in entry:
                    continue
                family = entry[platform]["family"]
                if family in seen:
                    self.fail(
                        f"Duplicate family {family!r} on {platform}: "
                        f"target {target_name!r} and {seen[family]!r}"
                    )
                seen[family] = target_name

    def test_required_fields_present(self):
        """Every platform entry must have the required fields."""
        required = {"family", "fetch-gfx-targets", "test-runs-on", "build_variants"}
        for target_name, entry in ALL_FAMILIES.items():
            for platform in ("linux", "windows"):
                if platform not in entry:
                    continue
                platform_info = entry[platform]
                missing = required - platform_info.keys()
                if missing:
                    self.fail(
                        f"{target_name}/{platform} missing required fields: {missing}"
                    )

    def test_build_variants_non_empty(self):
        """Every platform entry must list at least one build variant."""
        for target_name, entry in ALL_FAMILIES.items():
            for platform in ("linux", "windows"):
                if platform not in entry:
                    continue
                variants = entry[platform].get("build_variants", [])
                if not variants:
                    self.fail(f"{target_name}/{platform} has empty build_variants")


class TestExternalConfig(unittest.TestCase):
    """Tests for external config loading functionality."""

    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_load_external_runner_config_returns_none_when_env_not_set(self):
        """load_external_runner_config returns None when CI_CONFIG_PATH is not set."""
        if "CI_CONFIG_PATH" in os.environ:
            del os.environ["CI_CONFIG_PATH"]
        result = load_external_runner_config()
        self.assertIsNone(result)

    def test_load_external_runner_config_returns_none_when_env_empty(self):
        """load_external_runner_config returns None when CI_CONFIG_PATH is empty."""
        os.environ["CI_CONFIG_PATH"] = ""
        result = load_external_runner_config()
        self.assertIsNone(result)

    def test_load_external_runner_config_returns_none_when_import_fails(self):
        """load_external_runner_config returns None when ci_config_api import fails."""
        os.environ["CI_CONFIG_PATH"] = "/nonexistent/path"
        result = load_external_runner_config()
        self.assertIsNone(result)

    def test_get_all_families_overlays_external_runner_config(self):
        """get_all_families_for_trigger_types overlays external runner labels."""
        # External config provides runner labels that overlay local definitions
        fake_config = {
            "runner_labels": {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "external-runner-label",
                        "test-runs-on-multi-gpu": "external-multi-gpu-runner",
                    }
                }
            }
        }
        with mock.patch.object(
            amdgpu_family_matrix,
            "load_external_runner_config",
            return_value=fake_config,
        ):
            result = get_all_families_for_trigger_types(["presubmit"])

        # gfx94x should exist from local definitions
        self.assertIn("gfx94x", result)
        # Runner labels should be overlaid from external config
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on"], "external-runner-label"
        )
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on-multi-gpu"],
            "external-multi-gpu-runner",
        )
        # Non-runner fields should come from local definitions
        self.assertEqual(result["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")
        self.assertIn("release", result["gfx94x"]["linux"]["build_variants"])

    def test_get_all_families_falls_back_to_local_when_no_external_config(self):
        """get_all_families_for_trigger_types uses local matrix when no external config."""
        if "CI_CONFIG_PATH" in os.environ:
            del os.environ["CI_CONFIG_PATH"]
        result = get_all_families_for_trigger_types(["presubmit"])
        # Should contain entries from local presubmit matrix
        self.assertIn("gfx94x", result)

    def test_get_build_runner_labels_uses_external_config_when_available(self):
        """get_build_runner_labels uses external config when available."""
        fake_config = {
            "build_runners": {
                "linux": {"default": [{"label": "custom-runner", "weight": 1.0}]}
            }
        }
        with mock.patch.object(
            amdgpu_family_matrix,
            "load_external_runner_config",
            return_value=fake_config,
        ):
            result = get_build_runner_labels()
        self.assertEqual(result["linux"]["default"][0]["label"], "custom-runner")

    def test_get_build_runner_labels_falls_back_to_local_when_no_external_config(self):
        """get_build_runner_labels uses local config when no external config."""
        if "CI_CONFIG_PATH" in os.environ:
            del os.environ["CI_CONFIG_PATH"]
        result = get_build_runner_labels()
        # Should contain local BUILD_RUNNER_LABELS
        self.assertIn("linux", result)
        self.assertIn("default", result["linux"])

    def test_families_without_external_runners_still_buildable(self):
        """Families defined locally are buildable even without external runner config."""
        # External config with no runner_labels - simulates external config
        # that only has build_runners but no test runner config for a family
        fake_config = {
            "build_runners": {
                "linux": {"default": [{"label": "build-runner", "weight": 1.0}]}
            },
            "runner_labels": {},  # No test runners configured
        }
        with mock.patch.object(
            amdgpu_family_matrix,
            "load_external_runner_config",
            return_value=fake_config,
        ):
            result = get_all_families_for_trigger_types(["presubmit"])

        # gfx94x should still exist from local definitions
        self.assertIn("gfx94x", result)
        # Should have local runner label (not overlaid)
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on"],
            "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
        )
        # Build variants should still be defined
        self.assertIn("release", result["gfx94x"]["linux"]["build_variants"])

    def test_runner_labels_overlays_all_keys(self):
        """runner_labels section overlays all its keys onto local definitions."""
        fake_config = {
            "runner_labels": {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "external-runner",
                        "test-runs-on-multi-gpu": "external-multi-gpu",
                        "custom-runner-key": "custom-value",
                    }
                }
            }
        }
        with mock.patch.object(
            amdgpu_family_matrix,
            "load_external_runner_config",
            return_value=fake_config,
        ):
            result = get_all_families_for_trigger_types(["presubmit"])

        # All keys from runner_labels should be overlaid
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "external-runner")
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on-multi-gpu"], "external-multi-gpu"
        )
        self.assertEqual(result["gfx94x"]["linux"]["custom-runner-key"], "custom-value")
        # Local build config should still be present (not in runner_labels)
        self.assertEqual(result["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")
        self.assertIn("release", result["gfx94x"]["linux"]["build_variants"])

    def test_v1_external_config_extracts_runner_labels(self):
        """V1 config format (gpu_families) is handled for backward compatibility."""
        # V1 format has gpu_families organized by trigger type
        fake_v1_config = {
            "gpu_families": {
                "presubmit": {
                    "gfx94x": {
                        "linux": {
                            "test-runs-on": "v1-runner-label",
                            "test-runs-on-multi-gpu": "v1-multi-gpu-runner",
                            "family": "gfx94X-dcgpu",  # Should be ignored
                            "build_variants": ["release"],  # Should be ignored
                        }
                    }
                }
            }
        }
        with mock.patch.object(
            amdgpu_family_matrix,
            "load_external_runner_config",
            return_value=fake_v1_config,
        ):
            result = get_all_families_for_trigger_types(["presubmit"])

        # Runner labels should be extracted and overlaid
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "v1-runner-label")
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on-multi-gpu"], "v1-multi-gpu-runner"
        )
        # Non-runner keys should come from local definitions
        self.assertEqual(result["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")
        self.assertIn("asan", result["gfx94x"]["linux"]["build_variants"])

    def test_load_external_runner_config_v2_api_success(self):
        """load_external_runner_config successfully calls load_config(version=2)."""
        # Create a mock config object that mimics the v2 API
        mock_config = mock.MagicMock()
        mock_config.get_gpu_runner_labels.return_value = {
            "gfx94x": {
                "linux": {"test-runs-on": "v2-runner-label"},
            }
        }
        mock_config.build_runners = {
            "linux": {"default": [{"label": "v2-build-runner", "weight": 1.0}]}
        }

        # Create mock ci_config_api module
        mock_ci_config_api = mock.MagicMock()
        mock_ci_config_api.load_config.return_value = mock_config

        os.environ["CI_CONFIG_PATH"] = "/fake/config/path"

        with mock.patch.dict(sys.modules, {"ci_config_api": mock_ci_config_api}):
            # Need to reimport to pick up the mocked module
            import importlib

            importlib.reload(amdgpu_family_matrix)
            result = amdgpu_family_matrix.load_external_runner_config()

        # Verify load_config was called with version=2
        mock_ci_config_api.load_config.assert_called_once()
        call_kwargs = mock_ci_config_api.load_config.call_args
        self.assertEqual(call_kwargs.kwargs.get("version"), 2)

        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertIn("runner_labels", result)
        self.assertIn("build_runners", result)
        self.assertEqual(
            result["runner_labels"]["gfx94x"]["linux"]["test-runs-on"],
            "v2-runner-label",
        )
        self.assertEqual(
            result["build_runners"]["linux"]["default"][0]["label"],
            "v2-build-runner",
        )


# ---------------------------------------------------------------------------
# Build runner selection
# ---------------------------------------------------------------------------


class TestBuildRunnerSelection(unittest.TestCase):
    """Tests for select_build_runner() in amdgpu_family_matrix.py.

    CI_CONFIG_PATH is cleared to ensure tests use local definitions only.
    """

    def setUp(self):
        self._orig_env = os.environ.copy()
        if "CI_CONFIG_PATH" in os.environ:
            del os.environ["CI_CONFIG_PATH"]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_select_build_runner(self):
        """select_build_runner() returns the correct label for each platform/variant/size."""
        cases = [
            # (platform, variant, size, expected_runner_label)
            ("linux", "release", "large", "aws-linux-scale-rocm-prod"),
            ("windows", "release", "large", "azure-windows-scale-rocm"),
            # Sanitizer builds always use the large runner regardless of requested size
            ("linux", "asan", "small", "aws-linux-scale-rocm-large"),
            ("linux", "tsan", "medium", "aws-linux-scale-rocm-large"),
            ("linux", "release", "small", "aws-linux-scale-rocm-small"),
            # Windows has no small/medium pool — falls back to the Windows default
            ("windows", "release", "small", "azure-windows-scale-rocm"),
            ("linux", "release", "medium", "aws-linux-scale-rocm-medium"),
            ("windows", "release", "medium", "azure-windows-scale-rocm"),
        ]
        with patch("random.random", return_value=0.5):
            for platform, variant, size, expected in cases:
                with self.subTest(platform=platform, variant=variant, size=size):
                    self.assertEqual(
                        select_build_runner(platform, variant, size=size),
                        expected,
                    )


class TestBuildVariantTestTriggers(unittest.TestCase):
    """Trigger policy is data, so adding a variant or an event is an edit here
    rather than a new workflow input.

    Goes through the module rather than imported names: an earlier test in this
    file reloads amdgpu_family_matrix, which rebinds its globals.
    """

    @property
    def _triggers(self):
        return amdgpu_family_matrix.build_variant_test_triggers

    def _runs(self, *args, **kwargs):
        return amdgpu_family_matrix.build_variant_runs_tests(*args, **kwargs)

    def test_variants_without_a_policy_always_test(self):
        """release is not in the table and must stay unaffected."""
        for event in ["pull_request", "push", "schedule", "workflow_dispatch"]:
            with self.subTest(event=event):
                self.assertTrue(self._runs("release", event))

    def test_host_asan_still_tests_on_nightly_triggers(self):
        for event in ["schedule", "workflow_dispatch"]:
            with self.subTest(event=event):
                self.assertTrue(self._runs("host-asan", event))

    def test_host_asan_still_skips_postsubmit(self):
        self.assertFalse(self._runs("host-asan", "push"))

    def test_host_asan_tests_on_presubmit_without_a_label(self):
        """ROCm/TheRock#7202 requires this to run with no opt-in label."""
        self.assertTrue(self._runs("host-asan", "pull_request"))

    def test_debug_variant_follows_the_same_policy(self):
        """The gate matches on the host-asan prefix, so both forms need a rule."""
        self.assertFalse(self._runs("host-asan-debug", "push"))
        self.assertTrue(self._runs("host-asan-debug", "schedule"))

    def test_an_event_with_no_rule_does_not_test(self):
        self.assertFalse(self._runs("host-asan", "repository_dispatch"))

    def test_a_malformed_rule_raises(self):
        """A typo should fail the configure step, not quietly disable tests."""
        with mock.patch.dict(self._triggers, {"bogus": {"push": "enabld"}}):
            with self.assertRaisesRegex(ValueError, "expected 'enabled'"):
                self._runs("bogus", "push")

    def test_adding_postsubmit_needs_no_new_input(self):
        """The design requirement from the #7780 review."""
        with mock.patch.dict(
            self._triggers,
            {"host-asan": {**self._triggers["host-asan"], "push": "enabled"}},
        ):
            self.assertTrue(self._runs("host-asan", "push"))


if __name__ == "__main__":
    unittest.main()
