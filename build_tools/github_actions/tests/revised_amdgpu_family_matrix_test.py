# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for revised_amdgpu_family_matrix.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from revised_amdgpu_family_matrix import (
    GpuTarget,
    PlatformConfig,
    RunnerConfig,
    Trigger,
    get_all_targets,
    get_target,
    get_targets_for_trigger,
    to_legacy_matrix,
)


class TestGpuTarget(unittest.TestCase):
    """Tests for GpuTarget dataclass."""

    def test_default_build_targets(self):
        """Build targets default to [target]."""
        target = GpuTarget(target="gfx942")
        self.assertEqual(target.build_targets, ["gfx942"])

    def test_explicit_build_targets(self):
        """Explicit build targets are preserved."""
        target = GpuTarget(target="gfx942", build_targets=["gfx942", "gfx942:xnack+"])
        self.assertEqual(target.build_targets, ["gfx942", "gfx942:xnack+"])

    def test_supports_trigger_linux(self):
        """Check trigger support on Linux."""
        target = GpuTarget(
            target="gfx942",
            linux=PlatformConfig(triggers={Trigger.PRESUBMIT, Trigger.NIGHTLY}),
        )
        self.assertTrue(target.supports_trigger(Trigger.PRESUBMIT, "linux"))
        self.assertTrue(target.supports_trigger(Trigger.NIGHTLY, "linux"))
        self.assertFalse(target.supports_trigger(Trigger.POSTSUBMIT, "linux"))
        self.assertFalse(target.supports_trigger(Trigger.PRESUBMIT, "windows"))

    def test_supports_trigger_no_platform(self):
        """No platform config means no trigger support."""
        target = GpuTarget(target="gfx942", linux=None, windows=None)
        self.assertFalse(target.supports_trigger(Trigger.PRESUBMIT, "linux"))

    def test_to_dict_single_platform(self):
        """Serialize single platform config."""
        target = GpuTarget(
            target="gfx942",
            linux=PlatformConfig(
                triggers={Trigger.PRESUBMIT},
                build_variants=["release", "asan"],
                runners=RunnerConfig(test="linux-gfx942-gpu"),
            ),
        )
        result = target.to_dict("linux")
        self.assertEqual(result["family"], "gfx942")
        self.assertEqual(result["fetch-gfx-targets"], ["gfx942"])
        self.assertEqual(result["build_variants"], ["release", "asan"])
        self.assertEqual(result["test-runs-on"], "linux-gfx942-gpu")

    def test_to_dict_both_platforms(self):
        """Serialize both platforms."""
        target = GpuTarget(
            target="gfx1151",
            linux=PlatformConfig(runners=RunnerConfig(test="linux-runner")),
            windows=PlatformConfig(runners=RunnerConfig(test="windows-runner")),
        )
        result = target.to_dict()
        self.assertIn("linux", result)
        self.assertIn("windows", result)
        self.assertEqual(result["linux"]["test-runs-on"], "linux-runner")
        self.assertEqual(result["windows"]["test-runs-on"], "windows-runner")


class TestPlatformConfig(unittest.TestCase):
    """Tests for PlatformConfig dataclass."""

    def test_default_triggers(self):
        """Default trigger is PRESUBMIT."""
        config = PlatformConfig()
        self.assertEqual(config.triggers, {Trigger.PRESUBMIT})

    def test_default_build_variants(self):
        """Default build variant is release."""
        config = PlatformConfig()
        self.assertEqual(config.build_variants, ["release"])

    def test_to_dict_bypass_tests(self):
        """bypass_tests_for_releases flag is serialized."""
        config = PlatformConfig(bypass_tests_for_releases=True)
        result = config.to_dict("gfx1100")
        self.assertTrue(result.get("bypass_tests_for_releases"))

    def test_to_dict_nightly_only(self):
        """nightly_check_only_for_family when only nightly trigger."""
        config = PlatformConfig(triggers={Trigger.NIGHTLY})
        result = config.to_dict("gfx1100")
        self.assertTrue(result.get("nightly_check_only_for_family"))

    def test_to_dict_presubmit_no_nightly_flag(self):
        """No nightly flag when presubmit is included."""
        config = PlatformConfig(triggers={Trigger.PRESUBMIT, Trigger.NIGHTLY})
        result = config.to_dict("gfx942")
        self.assertNotIn("nightly_check_only_for_family", result)


class TestRegistry(unittest.TestCase):
    """Tests for GPU target registry."""

    def test_get_target_case_insensitive(self):
        """Target lookup is case-insensitive."""
        target = get_target("GFX942")
        self.assertIsNotNone(target)
        self.assertEqual(target.target, "gfx942")

        target2 = get_target("gfx942")
        self.assertIsNotNone(target2)
        self.assertEqual(target.target, target2.target)

    def test_get_target_unknown(self):
        """Unknown target returns None."""
        target = get_target("gfx_unknown")
        self.assertIsNone(target)

    def test_get_all_targets(self):
        """All targets are registered."""
        targets = get_all_targets()
        self.assertGreater(len(targets), 0)
        names = [t.target for t in targets]
        self.assertIn("gfx942", names)
        self.assertIn("gfx1151", names)

    def test_get_targets_for_trigger_presubmit_linux(self):
        """Get presubmit targets for Linux."""
        targets = get_targets_for_trigger(Trigger.PRESUBMIT, "linux")
        names = [t.target for t in targets]
        self.assertIn("gfx942", names)
        # gfx950 is postsubmit only
        self.assertNotIn("gfx950", names)

    def test_get_targets_for_trigger_postsubmit_linux(self):
        """Get postsubmit targets for Linux."""
        targets = get_targets_for_trigger(Trigger.POSTSUBMIT, "linux")
        names = [t.target for t in targets]
        self.assertIn("gfx942", names)
        self.assertIn("gfx950", names)


class TestLegacyExport(unittest.TestCase):
    """Tests for legacy matrix export."""

    def test_to_legacy_matrix(self):
        """Export to legacy format."""
        matrix = to_legacy_matrix()
        self.assertIn("gfx942", matrix)
        self.assertIn("linux", matrix["gfx942"])

    def test_to_legacy_matrix_filtered(self):
        """Export filtered by trigger and platform."""
        matrix = to_legacy_matrix(trigger=Trigger.PRESUBMIT, platform="linux")
        # All returned entries should support presubmit on linux
        for target_name in matrix:
            target = get_target(target_name)
            self.assertTrue(target.supports_trigger(Trigger.PRESUBMIT, "linux"))


if __name__ == "__main__":
    unittest.main()
