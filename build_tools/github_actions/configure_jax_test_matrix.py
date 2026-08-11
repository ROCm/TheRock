#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure JAX wheel test jobs for a GPU family.

The suite splits in two by the "multiaccelerator" pytest marker. Only that
subset needs more than one GPU, and it is a small part of the run, so the two go
to different runners: the family's 1-GPU runner takes the suite, which on one
GPU is the single-accelerator tests, and its multi-GPU runner takes the
multi-accelerator script.

Multi-GPU runners are scarce, so the second job is only worth its queue slot
when full testing is asked for. Short testing, which is what a pull request
gets, runs the suite on the 1-GPU runner alone, and a nightly asks for full
testing one day a week rather than every night.
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.amdgpu_family_matrix import get_all_families_for_trigger_types
from github_actions.configure_jax_release_matrix import RELEASE_TYPES
from github_actions.github_actions_api import gha_set_output

# Values of --test-scope, and the release types that pick them when the scope is
# left at "auto". CI and dev builds run per change, where an 8-GPU queue slot per
# job is too much, so they stay short.
SCOPE_SHORT = "short"
SCOPE_FULL = "full"
SCOPE_AUTO = "auto"
TEST_SCOPES = [SCOPE_AUTO, SCOPE_SHORT, SCOPE_FULL]

# A prerelease is worth a slot in the multi-GPU pool every time. A nightly takes
# one on this weekday (UTC, Monday being 0), because the pool is small and that
# subset moves slowly. Sunday is the quietest day for it.
ALWAYS_FULL_RELEASE_TYPES = ["prerelease"]
WEEKLY_FULL_RELEASE_TYPES = ["nightly"]
WEEKLY_FULL_WEEKDAY = 6

# --test-subset of run_jax_tests.py, which is which ROCm/jax suite script runs.
SUBSET_ALL = "all"
SUBSET_MULTI = "multi"


def platform_entry(target: str, platform: str) -> dict | None:
    """The family matrix entry for a target, or None if nothing matches.

    Matches the inner "family" or the outer key, because workflows pass the
    former while a manually dispatched run may pass the latter.
    See https://github.com/ROCm/TheRock/issues/1097.
    """
    matrix = get_all_families_for_trigger_types(["presubmit", "postsubmit"])
    for key, info_for_key in matrix.items():
        platform_for_key = info_for_key.get(platform)
        if not platform_for_key:
            # Some AMDGPU families are only supported on certain platforms.
            continue
        if target == platform_for_key.get("family") or key in target.lower():
            return platform_for_key
    return None


def today_utc() -> date:
    """The day the run is happening, which the weekly rule below reads."""
    return datetime.now(timezone.utc).date()


def resolve_scope(test_scope: str, release_type: str, today: date) -> str:
    """Which subsets to run, from the scope asked for or the release type.

    An explicit scope wins, so a workflow or a person can ask for the
    multi-accelerator tests on any day.
    """
    if test_scope != SCOPE_AUTO:
        return test_scope
    if release_type in ALWAYS_FULL_RELEASE_TYPES:
        return SCOPE_FULL
    if release_type in WEEKLY_FULL_RELEASE_TYPES:
        return SCOPE_FULL if today.weekday() == WEEKLY_FULL_WEEKDAY else SCOPE_SHORT
    return SCOPE_SHORT


def build_test_matrix(
    *,
    target: str,
    platform: str,
    scope: str,
) -> dict[str, list[dict[str, str]]]:
    entry = platform_entry(target, platform)
    if entry is None:
        raise ValueError(f"No {platform} AMDGPU family entry found for {target!r}")

    include: list[dict[str, str]] = []

    single_runner = entry.get("test-runs-on")
    if single_runner:
        include.append({"test_subset": SUBSET_ALL, "test_runs_on": single_runner})
    else:
        print(f"No {platform} test runner for {target}, so no tests will run")

    if scope == SCOPE_FULL:
        multi_runner = entry.get("test-runs-on-multi-gpu")
        if multi_runner:
            include.append({"test_subset": SUBSET_MULTI, "test_runs_on": multi_runner})
        else:
            # Sending them to a 1-GPU runner would skip every one of them.
            print(
                f"No {platform} multi-GPU test runner for {target}, so the"
                " multi-accelerator tests will not run"
            )

    for job in include:
        print(f"Including {job['test_subset']} tests on {job['test_runs_on']}")
    return {"include": include}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        help="GPU family to test, e.g. gfx94X-dcgpu",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default="linux",
        help="Test platform (default: linux)",
    )
    parser.add_argument(
        "--test-scope",
        choices=TEST_SCOPES,
        default=SCOPE_AUTO,
        help="Which subsets to run; 'auto' reads --release-type",
    )
    parser.add_argument(
        "--release-type",
        default="dev",
        # Rejected rather than defaulted, because an unrecognized release type
        # would quietly drop the multi-accelerator job from a release run.
        choices=RELEASE_TYPES,
        help="Release type the build is for (default: dev)",
    )
    args = parser.parse_args(argv)

    today = today_utc()
    scope = resolve_scope(args.test_scope, args.release_type, today)
    print(
        f"Configuring {args.platform} JAX tests for {args.target}:"
        f" {scope} scope (release type {args.release_type},"
        f" {today} ({today:%A}) in UTC)"
    )
    matrix = build_test_matrix(
        target=args.target,
        platform=args.platform,
        scope=scope,
    )
    gha_set_output(
        {
            "enabled": str(bool(matrix["include"])).lower(),
            "matrix": json.dumps(matrix, separators=(",", ":")),
        }
    )


if __name__ == "__main__":
    main(sys.argv[1:])
