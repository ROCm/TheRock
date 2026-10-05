# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for test_controls.py."""

import unittest

from test_controls import (
    TestControls,
    get_test_controls,
    should_disable_tests,
    get_test_filter_override,
    should_skip_component,
    filter_components_by_controls,
    apply_test_filter_override,
)


class TestTestControls(unittest.TestCase):
    def test_default_values(self):
        controls = TestControls(family="gfx94x", platform="linux")
        self.assertTrue(controls.tests_enabled)
        self.assertEqual(controls.test_filter_override, "")
        self.assertEqual(controls.disabled_test_components, [])
        self.assertFalse(controls.has_active_controls)

    def test_has_active_controls_tests_disabled(self):
        controls = TestControls(family="gfx94x", platform="linux", tests_enabled=False)
        self.assertTrue(controls.has_active_controls)

    def test_has_active_controls_filter_override(self):
        controls = TestControls(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        self.assertTrue(controls.has_active_controls)

    def test_has_active_controls_disabled_components(self):
        controls = TestControls(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        self.assertTrue(controls.has_active_controls)

    def test_invalid_filter_override_ignored(self):
        controls = TestControls(
            family="gfx94x", platform="linux", test_filter_override="invalid"
        )
        self.assertEqual(controls.test_filter_override, "")


class TestGetTestControls(unittest.TestCase):
    def test_from_none_config(self):
        controls = get_test_controls("gfx94x", "linux", None)
        self.assertTrue(controls.tests_enabled)
        self.assertFalse(controls.has_active_controls)

    def test_from_empty_config(self):
        controls = get_test_controls("gfx94x", "linux", {})
        self.assertTrue(controls.tests_enabled)

    def test_extracts_tests_enabled(self):
        config = {"tests_enabled": False}
        controls = get_test_controls("gfx94x", "linux", config)
        self.assertFalse(controls.tests_enabled)

    def test_extracts_filter_override(self):
        config = {"test_filter_override": "quick"}
        controls = get_test_controls("gfx94x", "linux", config)
        self.assertEqual(controls.test_filter_override, "quick")

    def test_extracts_disabled_components(self):
        config = {"disabled_test_components": ["rocblas", "miopen"]}
        controls = get_test_controls("gfx94x", "linux", config)
        self.assertEqual(controls.disabled_test_components, ["rocblas", "miopen"])


class TestShouldDisableTests(unittest.TestCase):
    def test_enabled(self):
        controls = TestControls(family="gfx94x", platform="linux", tests_enabled=True)
        should_disable, reason = should_disable_tests(controls)
        self.assertFalse(should_disable)
        self.assertEqual(reason, "")

    def test_disabled(self):
        controls = TestControls(family="gfx94x", platform="linux", tests_enabled=False)
        should_disable, reason = should_disable_tests(controls)
        self.assertTrue(should_disable)
        self.assertIn("tests_enabled=false", reason)


class TestGetTestFilterOverride(unittest.TestCase):
    def test_no_override(self):
        controls = TestControls(family="gfx94x", platform="linux")
        self.assertIsNone(get_test_filter_override(controls))

    def test_has_override(self):
        controls = TestControls(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        self.assertEqual(get_test_filter_override(controls), "quick")


class TestShouldSkipComponent(unittest.TestCase):
    def test_no_disabled_components(self):
        controls = TestControls(family="gfx94x", platform="linux")
        should_skip, reason = should_skip_component(controls, "rocblas")
        self.assertFalse(should_skip)

    def test_component_not_in_list(self):
        controls = TestControls(
            family="gfx94x", platform="linux", disabled_test_components=["miopen"]
        )
        should_skip, reason = should_skip_component(controls, "rocblas")
        self.assertFalse(should_skip)

    def test_component_in_list(self):
        controls = TestControls(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        should_skip, reason = should_skip_component(controls, "rocblas")
        self.assertTrue(should_skip)
        self.assertIn("rocblas", reason)


class TestFilterComponentsByControls(unittest.TestCase):
    def test_no_filtering(self):
        controls = TestControls(family="gfx94x", platform="linux")
        components = [{"job_name": "rocblas"}, {"job_name": "miopen"}]
        result = filter_components_by_controls(controls, components)
        self.assertEqual(len(result), 2)

    def test_filters_disabled_components(self):
        controls = TestControls(
            family="gfx94x", platform="linux", disabled_test_components=["rocblas"]
        )
        components = [{"job_name": "rocblas"}, {"job_name": "miopen"}]
        result = filter_components_by_controls(controls, components)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["job_name"], "miopen")


class TestApplyTestFilterOverride(unittest.TestCase):
    def test_no_override(self):
        controls = TestControls(family="gfx94x", platform="linux")
        test_type, reason = apply_test_filter_override(controls, "standard", "default")
        self.assertEqual(test_type, "standard")
        self.assertEqual(reason, "default")

    def test_with_override(self):
        controls = TestControls(
            family="gfx94x", platform="linux", test_filter_override="quick"
        )
        test_type, reason = apply_test_filter_override(controls, "standard", "default")
        self.assertEqual(test_type, "quick")
        self.assertIn("test control", reason)


if __name__ == "__main__":
    unittest.main()
