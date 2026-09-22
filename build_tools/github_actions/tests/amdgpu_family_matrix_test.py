#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for data invariants in amdgpu_family_matrix.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

# Clear CI_CONFIG_PATH before importing to ensure tests use local definitions only.
# This prevents external config from affecting test results.
if "CI_CONFIG_PATH" in os.environ:
    del os.environ["CI_CONFIG_PATH"]

import amdgpu_family_matrix
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    get_build_runner_labels,
    is_asan,
    is_device_asan,
    load_external_runner_config,
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


class TestAsanVariantChecks(unittest.TestCase):
    """host-asan instruments host code only, so it needs the accommodations
    is_asan() gates but not the skips for instrumented device code."""

    ASAN_FAMILY = ["asan", "asan-debug", "host-asan", "host-asan-debug"]

    def _with_variant(self, variant):
        return mock.patch.dict(os.environ, {"BUILD_VARIANT": variant})

    def test_is_asan_matches_every_asan_variant(self):
        for variant in self.ASAN_FAMILY:
            with self.subTest(variant=variant), self._with_variant(variant):
                self.assertTrue(is_asan())

    def test_is_device_asan_excludes_the_host_only_variants(self):
        for variant in ["host-asan", "host-asan-debug"]:
            with self.subTest(variant=variant), self._with_variant(variant):
                self.assertFalse(is_device_asan())

    def test_is_device_asan_matches_the_device_variants(self):
        for variant in ["asan", "asan-debug"]:
            with self.subTest(variant=variant), self._with_variant(variant):
                self.assertTrue(is_device_asan())

    def test_neither_matches_a_release_build(self):
        with self._with_variant("release"):
            self.assertFalse(is_asan())
            self.assertFalse(is_device_asan())

    def test_neither_matches_an_unset_variant(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_asan())
            self.assertFalse(is_device_asan())


if __name__ == "__main__":
    unittest.main()
