# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emergency lever system for CI test queue management.

This module provides centralized access to emergency levers that can be toggled
in therock-ci-config to quickly remediate test queue issues. The levers are:

- tests_enabled: Boolean to completely disable tests for a family/platform
- test_filter_override: Force a specific test tier (quick/standard/comprehensive/full)
- disabled_test_components: List of test components to skip

Usage in configure_multi_arch_ci.py:
    from emergency_levers import get_emergency_levers, apply_levers_to_test_decision

Usage in fetch_test_configurations.py:
    from emergency_levers import get_emergency_levers, should_skip_component

The data lives in therock-ci-config/runner-config-v2.json under gpu_runner_labels.
This module only reads and interprets; all data ownership is in therock-ci-config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

# Valid test filter values (matches _VALID_TEST_FILTER_TYPES in configure_multi_arch_ci.py)
VALID_TEST_FILTERS = frozenset(["quick", "standard", "comprehensive", "full"])


@dataclass
class EmergencyLevers:
    """Emergency lever settings for a specific family/platform combination."""

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
                f"[EMERGENCY-LEVERS] Invalid test_filter_override '{self.test_filter_override}' "
                f"for {self.family}/{self.platform}. Valid values: {sorted(VALID_TEST_FILTERS)}. "
                f"Ignoring override."
            )
            self.test_filter_override = ""

    @property
    def has_active_levers(self) -> bool:
        """Return True if any emergency lever is actively modifying behavior."""
        return (
            not self.tests_enabled
            or bool(self.test_filter_override)
            or bool(self.disabled_test_components)
        )

    def log_active_levers(self) -> None:
        """Log which emergency levers are active for visibility."""
        if not self.has_active_levers:
            return

        logging.info(f"[EMERGENCY-LEVERS] Active levers for {self.family}/{self.platform}:")
        if not self.tests_enabled:
            logging.info("  - tests_enabled: FALSE (all tests disabled)")
        if self.test_filter_override:
            logging.info(f"  - test_filter_override: {self.test_filter_override}")
        if self.disabled_test_components:
            logging.info(f"  - disabled_test_components: {self.disabled_test_components}")


def get_emergency_levers(
    family: str,
    platform: str,
    platform_config: dict | None,
) -> EmergencyLevers:
    """Extract emergency lever settings from platform config.

    Args:
        family: GPU family name (e.g., "gfx94x")
        platform: Platform name ("linux" or "windows")
        platform_config: Platform-specific config dict from amdgpu_family_matrix
                        (already overlaid with external config)

    Returns:
        EmergencyLevers instance with settings for this family/platform
    """
    if platform_config is None:
        return EmergencyLevers(family=family, platform=platform)

    levers = EmergencyLevers(
        family=family,
        platform=platform,
        tests_enabled=platform_config.get("tests_enabled", True),
        test_filter_override=platform_config.get("test_filter_override", ""),
        disabled_test_components=platform_config.get("disabled_test_components", []),
    )

    # Log active levers for CI visibility
    levers.log_active_levers()

    return levers


def should_disable_tests(levers: EmergencyLevers) -> tuple[bool, str]:
    """Check if tests should be completely disabled.

    Returns:
        Tuple of (should_disable, reason)
    """
    if not levers.tests_enabled:
        return True, f"tests_enabled=false for {levers.family}/{levers.platform}"
    return False, ""


def get_test_filter_override(levers: EmergencyLevers) -> str | None:
    """Get the test filter override if set.

    Returns:
        The override filter string, or None if no override is set
    """
    if levers.test_filter_override:
        return levers.test_filter_override
    return None


def should_skip_component(levers: EmergencyLevers, component_name: str) -> tuple[bool, str]:
    """Check if a specific test component should be skipped.

    Args:
        levers: EmergencyLevers instance
        component_name: Name of the test component (e.g., "rocblas", "miopen")

    Returns:
        Tuple of (should_skip, reason)
    """
    if levers.disabled_test_components and component_name in levers.disabled_test_components:
        return True, f"component '{component_name}' in disabled_test_components for {levers.family}/{levers.platform}"
    return False, ""


# =============================================================================
# Higher-level integration functions for use in CI scripts
# =============================================================================


def apply_test_filter_override(
    levers: EmergencyLevers,
    current_test_type: str,
    current_reason: str,
) -> tuple[str, str]:
    """Apply test filter override if set, otherwise return current values.

    This is used in configure_multi_arch_ci.py after _determine_test_type().

    Args:
        levers: EmergencyLevers instance
        current_test_type: The test type determined by normal logic
        current_reason: The reason for the current test type

    Returns:
        Tuple of (test_type, reason) - possibly overridden
    """
    override = get_test_filter_override(levers)
    if override:
        return override, f"emergency lever override for {levers.family}/{levers.platform}"
    return current_test_type, current_reason


def filter_components_by_levers(
    levers: EmergencyLevers,
    components: list[dict],
) -> list[dict]:
    """Filter out components that are disabled by emergency levers.

    This is used in fetch_test_configurations.py after building the component list.

    Args:
        levers: EmergencyLevers instance
        components: List of component config dicts

    Returns:
        Filtered list with disabled components removed
    """
    if not levers.disabled_test_components:
        return components

    filtered = []
    for component in components:
        job_name = component.get("job_name", "")
        should_skip, reason = should_skip_component(levers, job_name)
        if should_skip:
            logging.info(f"[EMERGENCY-LEVERS] Skipping component: {reason}")
        else:
            filtered.append(component)

    return filtered
