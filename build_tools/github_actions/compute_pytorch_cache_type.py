#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Computes the effective compiler cache type for a PyTorch wheel build and
writes it to GITHUB_OUTPUT as `cache_type`.

sccache for PyTorch uses an S3 bucket reached via an OIDC-assumed IAM role
(`therock-ci`). Pull requests from forks (and runs in repositories other than
ROCm/TheRock) cannot assume that role -- GitHub does not grant OIDC tokens to
fork workflows -- so any attempt to use sccache there fails with an AWS
AccessDenied / could-not-assume-role error (see ROCm/TheRock#5737).

To keep a single, consistent decision instead of scattering fork checks across
several workflow `if:` conditions, this script downgrades `sccache` to `none`
for fork PRs / non-ROCm repositories. `ccache` and `none` pass through
unchanged.

Used by the `*_pytorch_wheels_ci.yml` workflows.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from github_actions_api import gha_set_output, is_current_run_pr_from_fork


def compute_cache_type(cache_type: str, github_repository: str) -> str:
    """Return the effective cache type, downgrading sccache to none on forks.

    Args:
        cache_type: Requested cache type ("sccache", "ccache", or "none").
        github_repository: e.g. "ROCm/TheRock".
    """
    if cache_type != "sccache":
        return cache_type

    if is_current_run_pr_from_fork() or github_repository != "ROCm/TheRock":
        print(
            "sccache is unavailable for fork PRs / external repositories "
            "(no OIDC access to the sccache IAM role); using cache_type=none."
        )
        return "none"

    return "sccache"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Compute the effective PyTorch build cache type."
    )
    parser.add_argument(
        "--cache-type",
        type=str,
        default=os.environ.get("CACHE_TYPE", "sccache"),
        help='Requested cache type: "sccache", "ccache", or "none".',
    )
    args = parser.parse_args(argv)

    github_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    effective = compute_cache_type(args.cache_type, github_repository)
    print(f"Requested cache_type={args.cache_type!r} -> effective={effective!r}")
    gha_set_output({"cache_type": effective})


if __name__ == "__main__":
    main()
