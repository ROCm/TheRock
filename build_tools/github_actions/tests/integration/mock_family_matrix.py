# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A frozen GPU family matrix for integration scenarios.

The real matrix in ``amdgpu_family_matrix.py`` changes whenever a gfx target is
added, retired, or moved between runner pools. Scenarios that asserted on real
family names would break on every one of those edits for reasons that have
nothing to do with the behavior under test.

So the pipeline scenarios run against this frozen matrix instead. It is shaped
to exercise the interesting axes of ``select_targets`` and
``_expand_build_config_for_platform`` rather than to mirror production:

    family            platforms      builds on                tests on
    ---------------------------------------------------------------------------
    mock-primary      linux+windows  all triggers             all triggers
    mock-postsubmit   linux          postsubmit+ (no PR)      postsubmit, nightly
    mock-nightly      linux          nightly only             nightly only
    mock-submodule    linux          all triggers             submodule_bump only
    mock-asan         linux          all triggers             all triggers
    mock-windows      windows        presubmit, postsubmit,   same
                                     nightly

``mock-asan`` is the only family carrying sanitizer variants and a sandbox
runner, so asan scenarios can assert on it without perturbing the others.

Invariants that keep the real matrix honest (every family's
``tests_on_trigger`` is a subset of its ``builds_on_trigger``, every testable
family has a runner label, ...) are checked separately against the *real*
matrix in ``family_matrix_contract_test.py``.

One limitation: stage-reuse scenarios cannot use these families. Reuse
resolves the artifact names a baseline run must contain via
``expand_amdgpu_families``, which validates against the production family list
and raises ``Cannot expand AMDGPU target family`` on a mock name. Those
scenarios set ``families: real``.
"""

import copy

ALL_TRIGGERS = ["presubmit", "postsubmit", "submodule_bump", "nightly"]

MOCK_FAMILIES: dict[str, dict] = {
    # The workhorse: present on both platforms, built and tested everywhere.
    # Carries the optional runner keys (multi-gpu, wsl, weighted labels) so
    # scenarios can assert they are plumbed into per_family_info.
    "mock-primary": {
        "linux": {
            "family": "mock-primary",
            "test-runs-on": "mock-linux-primary",
            "test-runs-on-labels": [{"label": "mock-linux-primary", "count": 1}],
            "test-runs-on-multi-gpu": "mock-linux-primary-8gpu",
            "test-runs-on-wsl": "mock-wsl-primary",
            "fetch-gfx-targets": ["gfx0000"],
            "build_variants": ["release"],
            "builds_on_trigger": list(ALL_TRIGGERS),
            "tests_on_trigger": list(ALL_TRIGGERS),
        },
        "windows": {
            "family": "mock-primary",
            "test-runs-on": "mock-windows-primary",
            "fetch-gfx-targets": ["gfx0000"],
            "build_variants": ["release"],
            "builds_on_trigger": list(ALL_TRIGGERS),
            "tests_on_trigger": list(ALL_TRIGGERS),
        },
    },
    # Not built on PRs at all. Proves presubmit is narrower than postsubmit.
    "mock-postsubmit": {
        "linux": {
            "family": "mock-postsubmit",
            "test-runs-on": "mock-linux-postsubmit",
            "fetch-gfx-targets": ["gfx0001"],
            "build_variants": ["release"],
            "builds_on_trigger": ["postsubmit", "submodule_bump", "nightly"],
            "tests_on_trigger": ["postsubmit", "nightly"],
        },
    },
    # Nightly-only on both axes. Should never appear in a PR or push build.
    "mock-nightly": {
        "linux": {
            "family": "mock-nightly",
            "test-runs-on": "mock-linux-nightly",
            "fetch-gfx-targets": ["gfx0002"],
            "build_variants": ["release"],
            "builds_on_trigger": ["nightly"],
            "tests_on_trigger": ["nightly"],
        },
    },
    # Built everywhere but only *tested* on a submodule bump. This is the
    # family that separates "built" from "tested" in scenario assertions.
    "mock-submodule": {
        "linux": {
            "family": "mock-submodule",
            "test-runs-on": "mock-linux-submodule",
            "fetch-gfx-targets": ["gfx0003"],
            "build_variants": ["release"],
            "builds_on_trigger": list(ALL_TRIGGERS),
            "tests_on_trigger": ["submodule_bump"],
        },
    },
    # The only family with sanitizer variants, so release scenarios are
    # unaffected by asan plumbing and vice versa.
    "mock-asan": {
        "linux": {
            "family": "mock-asan",
            "test-runs-on": "mock-linux-asan",
            "test-runs-on-sandbox": "mock-linux-asan-sandbox",
            "fetch-gfx-targets": ["gfx0004"],
            "build_variants": ["release", "asan", "host-asan"],
            "builds_on_trigger": list(ALL_TRIGGERS),
            "tests_on_trigger": list(ALL_TRIGGERS),
        },
    },
    # Windows-only. Confirms per-platform filtering drops it from Linux
    # selections even though trigger defaults are computed platform-agnostically.
    "mock-windows": {
        "windows": {
            "family": "mock-windows",
            "test-runs-on": "mock-windows-only",
            "fetch-gfx-targets": ["gfx0005"],
            "build_variants": ["release"],
            "builds_on_trigger": ["presubmit", "postsubmit", "nightly"],
            "tests_on_trigger": ["presubmit", "postsubmit", "nightly"],
        },
    },
}

# Families reachable only when explicitly named, mirroring the real matrix's
# explicit_only tier (workflow_dispatch can select them; "all" cannot).
MOCK_EXPLICIT_ONLY_FAMILIES: dict[str, dict] = {
    "mock-explicit": {
        "linux": {
            "family": "mock-explicit",
            "test-runs-on": "mock-linux-explicit",
            "fetch-gfx-targets": ["gfx0006"],
            "build_variants": ["release"],
            "builds_on_trigger": ["explicit_only"],
            "tests_on_trigger": ["explicit_only"],
        },
    },
}


def mock_get_all_families_for_trigger_types(trigger_types: list[str]) -> dict:
    """Stand-in for ``amdgpu_family_matrix.get_all_families_for_trigger_types``.

    Matches the real contract in the two ways callers depend on:

    1. An empty ``trigger_types`` returns *all* non-explicit families.
    2. The result is a deep copy. ``select_targets`` mutates what it gets back
       (``all_families.update(...)`` for workflow_dispatch), so returning a
       shared dict would leak state between scenarios.
    """
    if trigger_types and "explicit_only" in trigger_types:
        return copy.deepcopy(MOCK_EXPLICIT_ONLY_FAMILIES)

    if not trigger_types:
        return copy.deepcopy(MOCK_FAMILIES)

    wanted = set(trigger_types)
    return {
        name: copy.deepcopy(config)
        for name, config in MOCK_FAMILIES.items()
        if any(
            wanted & set(platform_info.get("builds_on_trigger", []))
            for platform_info in config.values()
        )
    }
