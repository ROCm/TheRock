#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure JAX wheel test jobs for a GPU family.

The suite splits in two by the "multiaccelerator" pytest marker. Only that
subset needs more than one GPU, and it is a small part of the run, so the two go
to different runners: the family's 1-GPU runner takes the suite, which on one
GPU is the single-accelerator tests, and its multi-GPU runner takes the
multi-accelerator script.

--test-size says how much of that a run is worth:

  * small: the PR-sized selection on the 1-GPU runner, and nothing else. This
    one blocks a pull request, so it is the one that has to stay short.
  * medium: the whole single-GPU suite nightly, plus the multi-accelerator job
    one day a week. Multi-GPU runners are scarce enough that a nightly 8-GPU
    slot is not worth what that subset finds.
  * large: both, every time, for a release or prerelease.

Which day the week falls on comes from the wall clock, as it does for the
PyTorch pipeline, so re-running a whole Sunday nightly on a later day gives
that day's matrix; dispatch a large run to get the multi-accelerator tests
back.

The end-to-end workloads run as a step of the multi-GPU job rather than a job
of their own, so they follow whatever this emits.
"""

import argparse
import calendar
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.amdgpu_family_matrix import get_all_families_for_trigger_types
from github_actions.github_actions_api import gha_set_output

# Values of --test-size, as the module docstring above describes them. A size
# decides whether the multi-GPU job runs, and what each job selects and how long
# it is given.
TEST_SIZE_SMALL = "small"
TEST_SIZE_MEDIUM = "medium"
TEST_SIZE_LARGE = "large"
TEST_SIZES = [TEST_SIZE_SMALL, TEST_SIZE_MEDIUM, TEST_SIZE_LARGE]

# The day a medium run also takes a multi-GPU runner, in UTC, Monday being 0.
# Sunday is the quietest for the shared 8-GPU pool.
WEEKLY_MULTI_GPU_WEEKDAY = 6

# --test-subset of run_jax_tests.py, which is which ROCm/jax suite script runs:
# "all" is ci/run_pytest_rocm.sh, which leaves out what the host has no GPUs for,
# and "multi" is ci/run_pytest_rocm_multi.sh, the multi-accelerator tests alone.
TEST_SUBSET_ALL = "all"
TEST_SUBSET_MULTI = "multi"

# Step budgets in minutes, emitted with each job so the test workflow reads one
# rather than deriving it. The single-accelerator suite runs 35-70 minutes
# across versions, so 60 left no headroom: one job reached 100% of the tests and
# was still killed. The other two are a fraction of that run.
TIMEOUT_MINUTES_ALL = 120
TIMEOUT_MINUTES_MULTI = 90
TIMEOUT_MINUTES_SMALL = 45

# The PR-sized selection, which exists for the single-accelerator subset only:
# there is no reduced form of the multi-accelerator script.
SMALL_TEST_LIST = "external-builds/jax/test_selection/small_tests.txt"


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
    """Today in UTC, for a caller that did not say when its run began."""
    return datetime.now(timezone.utc).date()


def wants_multi_gpu(size: str, today: date) -> bool:
    """Whether this run should also take a multi-GPU runner.

    A medium run is the nightly, so this is what makes the multi-accelerator
    tests weekly rather than nightly. Asking for large runs them whatever day it
    is, which is also how someone gets them on demand.
    """
    if size == TEST_SIZE_LARGE:
        return True
    return size == TEST_SIZE_MEDIUM and today.weekday() == WEEKLY_MULTI_GPU_WEEKDAY


def build_test_matrix(
    *,
    target: str,
    platform: str,
    size: str,
    today: date,
) -> dict[str, list[dict[str, str | int]]]:
    entry = platform_entry(target, platform)
    if entry is None:
        raise ValueError(f"No {platform} AMDGPU family entry found for {target!r}")

    include: list[dict[str, str | int]] = []

    single_runner = entry.get("test-runs-on")
    if single_runner:
        small = size == TEST_SIZE_SMALL
        include.append(
            {
                "test_subset": TEST_SUBSET_ALL,
                "test_runs_on": single_runner,
                "test_timeout_minutes": (
                    TIMEOUT_MINUTES_SMALL if small else TIMEOUT_MINUTES_ALL
                ),
                # Empty means the whole subset, which is every size but small.
                "test_list": SMALL_TEST_LIST if small else "",
            }
        )
    else:
        # A family with no hardware configured carries an empty label, so this is
        # a skip rather than an error. Annotated because everything below it then
        # runs a smaller share of the suite than the scope asked for.
        print(
            f"::warning::No {platform} test runner for {target}, so the"
            " single-accelerator tests will not run"
        )

    if wants_multi_gpu(size, today):
        multi_runner = entry.get("test-runs-on-multi-gpu")
        if multi_runner:
            include.append(
                {
                    "test_subset": TEST_SUBSET_MULTI,
                    "test_runs_on": multi_runner,
                    "test_timeout_minutes": TIMEOUT_MINUTES_MULTI,
                    "test_list": "",
                }
            )
        else:
            # Sending them to a 1-GPU runner would skip every one of them.
            print(
                f"::warning::No {platform} multi-GPU test runner for {target}, so"
                " the multi-accelerator tests will not run"
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
        "--test-size",
        # Rejected rather than defaulted, because an unrecognized size would
        # quietly drop the multi-accelerator job from a release run.
        choices=TEST_SIZES,
        required=True,
        help="How much of the suite this run is worth",
    )
    args = parser.parse_args(argv)

    today = today_utc()
    print(
        f"Configuring {args.platform} JAX tests for {args.target}:"
        f" {args.test_size} size, {today} ({today:%A}) in UTC"
    )
    if args.test_size == TEST_SIZE_MEDIUM and not wants_multi_gpu(
        args.test_size, today
    ):
        # Annotated so that a nightly re-run onto another day is a visible loss
        # of the week of multi-accelerator coverage rather than a silent one.
        print(
            "::notice::The multi-accelerator tests run on"
            f" {calendar.day_name[WEEKLY_MULTI_GPU_WEEKDAY]}; dispatch this"
            " workflow with test_size large to run them today"
        )
    matrix = build_test_matrix(
        target=args.target,
        platform=args.platform,
        size=args.test_size,
        today=today,
    )
    gha_set_output(
        {
            "enabled": str(bool(matrix["include"])).lower(),
            "matrix": json.dumps(matrix, separators=(",", ":")),
        }
    )


if __name__ == "__main__":
    main(sys.argv[1:])
