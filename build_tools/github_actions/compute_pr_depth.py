#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compute the smallest `actions/checkout` fetch-depth for the current event."""

import sys
from typing import Any

from github_actions_api import gha_load_github_event, gha_set_output


def compute_fetch_depth(payload: dict[str, Any]) -> str:
    pr = payload.get("pull_request")
    if not isinstance(pr, dict):
        return "0"

    commits = pr.get("commits")
    if not isinstance(commits, int) or commits <= 0:
        return "0"

    return str(commits + 1)


def main(argv: list[str]) -> int:
    value = compute_fetch_depth(gha_load_github_event())
    print(f"fetch-depth = {value}")
    gha_set_output({"value": value})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
