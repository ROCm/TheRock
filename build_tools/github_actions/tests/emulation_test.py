# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the emulated test scripts under test_executable_scripts/."""

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))

import fetch_test_configurations

# Scripts under test_executable_scripts/ call logging.basicConfig() at import,
# by convention across that whole directory. That is right for a script run as
# `python test_foo.py` and wrong here: importing them would otherwise leave the
# root logger at INFO with a stderr handler for every *other* module in the
# `pytest build_tools` session, which is a collection-order dependency.
_root_logger = logging.getLogger()
_prior_level = _root_logger.level
_prior_handlers = list(_root_logger.handlers)

import emulation
import test_emulation_smoke

_root_logger.setLevel(_prior_level)
_root_logger.handlers[:] = _prior_handlers


class EmulationEnvTest(unittest.TestCase):
    """TEST_EMULATOR / TEST_EMULATOR_PROFILE reading."""

    def test_unset_is_not_emulated(self):
        self.assertFalse(emulation.is_emulated({}))
        self.assertEqual(emulation.emulator_name({}), "")
        self.assertEqual(emulation.emulator_profile({}), "")

    def test_blank_is_not_emulated(self):
        # GitHub Actions renders an unset matrix field as an empty string.
        env = {"TEST_EMULATOR": "", "TEST_EMULATOR_PROFILE": ""}
        self.assertFalse(emulation.is_emulated(env))

    def test_values_are_stripped(self):
        env = {"TEST_EMULATOR": " rocjitsu ", "TEST_EMULATOR_PROFILE": " mi350x\n"}
        self.assertTrue(emulation.is_emulated(env))
        self.assertEqual(emulation.emulator_name(env), "rocjitsu")
        self.assertEqual(emulation.emulator_profile(env), "mi350x")


class RocmPathTest(unittest.TestCase):
    def test_derived_from_therock_bin_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"THEROCK_BIN_DIR": str(Path(tmp) / "bin")}
            self.assertEqual(emulation.rocm_path(env), Path(tmp).resolve())

    def test_missing_bin_dir_raises(self):
        with self.assertRaises(RuntimeError):
            emulation.rocm_path({})


class EmulationSmokeTest(unittest.TestCase):
    """rocminfo output validation for test_emulation_smoke.py."""

    GOOD_MI350X = """
    Agent 1
      Name:  AMD EPYC
      Device Type: CPU
    Agent 2
      Name:  gfx950
      Device Type: GPU
    """

    def test_matching_agent_passes(self):
        self.assertEqual(
            test_emulation_smoke.check_rocminfo_output(self.GOOD_MI350X, "mi350x"), []
        )

    def test_wrong_agent_is_reported(self):
        problems = test_emulation_smoke.check_rocminfo_output(
            self.GOOD_MI350X, "mi450x"
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("expected a gfx1250 agent", problems[0])

    def test_no_agents_is_reported(self):
        problems = test_emulation_smoke.check_rocminfo_output("", "mi350x")
        self.assertIn("no agents at all", problems[0])

    def test_unknown_profile_only_checks_for_an_agent(self):
        # An unmapped profile should not fail the smoke test outright; it just
        # loses the agent-identity assertion.
        self.assertEqual(
            test_emulation_smoke.check_rocminfo_output(self.GOOD_MI350X, "future-gpu"),
            [],
        )

    def test_every_scheduled_profile_has_an_expected_agent(self):
        # fetch_test_configurations.py picks the profile from the AMDGPU
        # family; this map turns it back into the gfx target the emulated
        # agent must report. A profile missing here silently downgrades the
        # smoke test to "some agent exists", which is the one thing it is not
        # supposed to settle for.
        for profile in fetch_test_configurations._EMULATED_PROFILES:
            self.assertIn(profile, test_emulation_smoke.EXPECTED_GFX_BY_PROFILE)

    def test_expected_agent_matches_the_family_it_is_scheduled_for(self):
        # The two maps are written from opposite ends -- family -> profile in
        # fetch_test_configurations.py, profile -> gfx target here -- so a typo
        # in either shows up as a round trip that does not close. Compared as a
        # prefix because the family key is a family label ("gfx125") and the
        # target is a specific chip ("gfx1250").
        prefixes = fetch_test_configurations._MIRAGE_PROFILE_BY_FAMILY_PREFIX
        for family_prefix, profile in prefixes.items():
            with self.subTest(profile=profile):
                gfx_target = test_emulation_smoke.EXPECTED_GFX_BY_PROFILE[profile]
                self.assertTrue(
                    gfx_target.startswith(family_prefix),
                    f"family prefix {family_prefix} maps to profile {profile}, "
                    f"which is expected to present {gfx_target}",
                )


if __name__ == "__main__":
    unittest.main()
