# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for emergency_levers.py."""

import unittest

from emergency_levers import (
    EmergencyLevers,
    get_emergency_levers,
    should_disable_tests,
    get_test_filter_override,
    should_skip_component,
    filter_components_by_levers,
    apply_test_filter_override,
)


class TestEmergencyLevers(unittest.TestCase):
    def test_default_values(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux")
        self.assertTrue(levers.tests_enabled)
        self.assertEqual(levers.test_filter_override, "")
        self.assertEqual(levers.disabled_test_components, [])
        self.assertFalse(levers.has_active_levers)

    def test_has_active_levers_tests_disabled(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux", tests_enabled=False)
        self.assertTrue(levers.has_active_levers)

    def test_has_active_levers_filter_override(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        self.assertTrue(levers.has_active_levers)

    def test_has_active_levers_disabled_components(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        self.assertTrue(levers.has_active_levers)

    def test_invalid_filter_override_ignored(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", test_filter_override="invalid"
        )
        self.assertEqual(levers.test_filter_override, "")


class TestGetEmergencyLevers(unittest.TestCase):
    def test_from_none_config(self):
        levers = get_emergency_levers("gfx94x", "linux", None)
        self.assertTrue(levers.tests_enabled)
        self.assertFalse(levers.has_active_levers)

    def test_from_empty_config(self):
        levers = get_emergency_levers("gfx94x", "linux", {})
        self.assertTrue(levers.tests_enabled)

    def test_extracts_tests_enabled(self):
        config = {"tests_enabled": False}
        levers = get_emergency_levers("gfx94x", "linux", config)
        self.assertFalse(levers.tests_enabled)

    def test_extracts_filter_override(self):
        config = {"test_filter_override": "quick"}
        levers = get_emergency_levers("gfx94x", "linux", config)
        self.assertEqual(levers.test_filter_override, "quick")

    def test_extracts_disabled_components(self):
        config = {"disabled_test_components": ["rocblas", "miopen"]}
        levers = get_emergency_levers("gfx94x", "linux", config)
        self.assertEqual(levers.disabled_test_components, ["rocblas", "miopen"])


class TestShouldDisableTests(unittest.TestCase):
    def test_enabled(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux", tests_enabled=True)
        should_disable, reason = should_disable_tests(levers)
        self.assertFalse(should_disable)
        self.assertEqual(reason, "")

    def test_disabled(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux", tests_enabled=False)
        should_disable, reason = should_disable_tests(levers)
        self.assertTrue(should_disable)
        self.assertIn("tests_enabled=false", reason)


class TestGetTestFilterOverride(unittest.TestCase):
    def test_no_override(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux")
        self.assertIsNone(get_test_filter_override(levers))

    def test_has_override(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        self.assertEqual(get_test_filter_override(levers), "quick")


class TestShouldSkipComponent(unittest.TestCase):
    def test_no_disabled_components(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux")
        should_skip, reason = should_skip_component(levers, "rocblas")
        self.assertFalse(should_skip)

    def test_component_not_in_list(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", disabled_test_components=["miopen"]
        )
        should_skip, reason = should_skip_component(levers, "rocblas")
        self.assertFalse(should_skip)

    def test_component_in_list(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        should_skip, reason = should_skip_component(levers, "rocblas")
        self.assertTrue(should_skip)
        self.assertIn("rocblas", reason)


class TestFilterComponentsByLevers(unittest.TestCase):
    def test_no_filtering(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux")
        components = [{"job_name": "rocblas"}, {"job_name": "miopen"}]
        result = filter_components_by_levers(levers, components)
        self.assertEqual(len(result), 2)

    def test_filters_disabled_components(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        components = [{"job_name": "rocblas"}, {"job_name": "miopen"}]
        result = filter_components_by_levers(levers, components)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["job_name"], "miopen")


class TestApplyTestFilterOverride(unittest.TestCase):
    def test_no_override(self):
        levers = EmergencyLevers(family="gfx94x", platform="linux")
        test_type, reason = apply_test_filter_override(levers, "standard", "default")
        self.assertEqual(test_type, "standard")
        self.assertEqual(reason, "default")

    def test_with_override(self):
        levers = EmergencyLevers(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        test_type, reason = apply_test_filter_override(levers, "standard", "default")
        self.assertEqual(test_type, "quick")
        self.assertIn("emergency lever", reason)


if __name__ == "__main__":
    unittest.main()
