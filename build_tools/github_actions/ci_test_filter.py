# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse `test_filter:<type>` labels into a test type.

Standard library only: configure_external_repo_ci.py imports this module from
external repos that check out only build_tools/github_actions.
See docs/development/test_filtering.md for the test types.
"""

from typing import Iterable, Optional

VALID_TEST_FILTER_TYPES = {
    "quick",
    "standard",
    "comprehensive",
    "full",
    # Opt-in suites for components that declare "workloads_timeout_minutes"
    # in fetch_test_configurations.py; only selectable via test_filter:.
    "workloads",
}


def parse_test_filter_label(labels: Iterable[str]) -> Optional[str]:
    """Return the type of the first `test_filter:` label, or None if none.

    Raises ValueError for an unrecognized type, so a typo fails the run instead
    of silently running the default test type.
    """
    for label in labels:
        if not label.startswith("test_filter:"):
            continue
        filter_type = label.split(":")[1]
        if filter_type not in VALID_TEST_FILTER_TYPES:
            raise ValueError(
                f"Unrecognized test_filter value: {filter_type!r}. "
                f"Valid values: {sorted(VALID_TEST_FILTER_TYPES)}"
            )
        return filter_type
    return None
