# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test control system for CI configuration.

This module provides centralized access to test controls that can be configured
in therock-ci-config to manage test behavior per gfx target. The controls are:

- tests_enabled: Boolean to completely disable tests for a family/platform
- test_filter_override: Force a specific test tier (quick/standard/comprehensive/full)
- disabled_test_components: List of test components to skip

Use cases:
- Emergency response: Quickly reduce test load when runner capacity is constrained
- Steady-state configuration: Selectively enable/disable tests for specific targets

Usage in configure_multi_arch_ci.py:
    from test_controls import get_test_controls, apply_test_filter_override

Usage in fetch_test_configurations.py:
    from test_controls import get_test_controls, should_skip_component

The data lives in therock-ci-config/runner-config-v2.json under gpu_runner_labels.
This module only reads and interprets; all data ownership is in therock-ci-config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# Valid test filter values (matches _VALID_TEST_FILTER_TYPES in configure_multi_arch_ci.py)
VALID_TEST_FILTERS = frozenset(["quick", "standard", "comprehensive", "full"])


@dataclass
class TestControls:
    """Test control settings for a specific family/platform combination."""

    family: str
    platform: str
    tests_enabled: bool = True
    test_filter_override: str = ""
    disabled_test_components: list[str] | None = None

    def __post_init__(self):
        if self.disabled_test_components is None:
            self.disabled_test_components = []

        # Validate test_filter_override if set
        if self.test_filter_override and self.test_filter_override not in VALID_TEST_FILTERS:
            logging.warning(
                f"[TEST-CONTROLS] Invalid test_filter_override '{self.test_filter_override}' "
                f"for {self.family}/{self.platform}. Valid values: {sorted(VALID_TEST_FILTERS)}. "
                f"Ignoring override."
            )
            self.test_filter_override = ""

    @property
    def has_active_controls(self) -> bool:
        """Return True if any test control is actively modifying behavior."""
        return (
            not self.tests_enabled
            or bool(self.test_filter_override)
            or bool(self.disabled_test_components)
        )

    def log_active_controls(self) -> None:
        """Log which test controls are active for visibility."""
        if not self.has_active_controls:
            return

        logging.info(f"[TEST-CONTROLS] Active controls for {self.family}/{self.platform}:")
        if not self.tests_enabled:
            logging.info("  - tests_enabled: FALSE (all tests disabled)")
        if self.test_filter_override:
            logging.info(f"  - test_filter_override: {self.test_filter_override}")
        if self.disabled_test_components:
            logging.info(f"  - disabled_test_components: {self.disabled_test_components}")


def get_test_controls(
    family: str,
    platform: str,
    platform_config: dict | None,
) -> TestControls:
    """Extract test control settings from platform config.

    Args:
        family: GPU family name (e.g., "gfx94x")
        platform: Platform name ("linux" or "windows")
        platform_config: Platform-specific config dict from amdgpu_family_matrix
                        (already overlaid with external config)

    Returns:
        TestControls instance with settings for this family/platform
    """
    if platform_config is None:
        return TestControls(family=family, platform=platform)

    controls = TestControls(
        family=family,
        platform=platform,
        tests_enabled=platform_config.get("tests_enabled", True),
        test_filter_override=platform_config.get("test_filter_override", ""),
        disabled_test_components=platform_config.get("disabled_test_components", []),
    )

    # Log active controls for CI visibility
    controls.log_active_controls()

    return controls


def should_disable_tests(controls: TestControls) -> tuple[bool, str]:
    """Check if tests should be completely disabled.

    Returns:
        Tuple of (should_disable, reason)
    """
    if not controls.tests_enabled:
        return True, f"tests_enabled=false for {controls.family}/{controls.platform}"
    return False, ""


def get_test_filter_override(controls: TestControls) -> str | None:
    """Get the test filter override if set.

    Returns:
        The override filter string, or None if no override is set
    """
    if controls.test_filter_override:
        return controls.test_filter_override
    return None


def should_skip_component(controls: TestControls, component_name: str) -> tuple[bool, str]:
    """Check if a specific test component should be skipped.

    Args:
        controls: TestControls instance
        component_name: Name of the test component (e.g., "rocblas", "miopen")

    Returns:
        Tuple of (should_skip, reason)
    """
    if controls.disabled_test_components and component_name in controls.disabled_test_components:
        return True, f"component '{component_name}' in disabled_test_components for {controls.family}/{controls.platform}"
    return False, ""


# =============================================================================
# Higher-level integration functions for use in CI scripts
# =============================================================================


def apply_test_filter_override(
    controls: TestControls,
    current_test_type: str,
    current_reason: str,
) -> tuple[str, str]:
    """Apply test filter override if set, otherwise return current values.

    This is used in configure_multi_arch_ci.py after _determine_test_type().

    Args:
        controls: TestControls instance
        current_test_type: The test type determined by normal logic
        current_reason: The reason for the current test type

    Returns:
        Tuple of (test_type, reason) - possibly overridden
    """
    override = get_test_filter_override(controls)
    if override:
        return override, f"test control override for {controls.family}/{controls.platform}"
    return current_test_type, current_reason


def filter_components_by_controls(
    controls: TestControls,
    components: list[dict],
) -> list[dict]:
    """Filter out components that are disabled by test controls.

    This is used in fetch_test_configurations.py after building the component list.

    Args:
        controls: TestControls instance
        components: List of component config dicts

    Returns:
        Filtered list with disabled components removed
    """
    if not controls.disabled_test_components:
        return components

    filtered = []
    for component in components:
        job_name = component.get("job_name", "")
        should_skip, reason = should_skip_component(controls, job_name)
        if should_skip:
            logging.info(f"[TEST-CONTROLS] Skipping component: {reason}")
        else:
            filtered.append(component)

    return filtered
