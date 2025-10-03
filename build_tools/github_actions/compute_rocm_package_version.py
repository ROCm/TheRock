#!/usr/bin/env python

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Computes a ROCm package version with an appropriate suffix for a given release type.

Writes to 'version' in GITHUB_OUTPUT.
"""

import argparse
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

from github_actions_utils import *

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def load_rocm_version() -> str:
    """Loads the rocm-version from the repository's version.json file."""
    version_file = THEROCK_DIR / "version.json"
    _log(f"Loading version from file '{version_file.resolve()}'")
    with open(version_file, "rt") as f:
        loaded_file = json.load(f)
        return loaded_file["rocm-version"]


def get_git_sha():
    """Gets the current git SHA, either from GITHUB_SHA or running git commands."""

    # https://docs.github.com/en/actions/reference/workflows-and-actions/variables
    github_sha = os.getenv("GITHUB_SHA")

    if github_sha:
        git_sha = github_sha
    else:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=THEROCK_DIR,
            text=True,
        ).strip()

    # Note: we could shorten the sha to 8 characters if we wanted here.
    return git_sha


def get_current_date():
    """Gets the current date as YYYYMMDD."""
    return datetime.today().strftime("%Y%m%d")


def compute_version(
    release_type: str | None,
    custom_version_suffix: str | None,
    override_base_version: str | None,
) -> str:
    if override_base_version:
        base_version = override_base_version
    else:
        base_version = load_rocm_version()
    _log(f"Base version  : '{base_version}'")

    if custom_version_suffix:
        # Trust the custom suffix to satify the general rules:
        # https://packaging.python.org/en/latest/specifications/version-specifiers/
        version_suffix = custom_version_suffix
    elif release_type == "dev":
        # Construct a dev release version:
        # https://packaging.python.org/en/latest/specifications/version-specifiers/#developmental-releases
        git_sha = get_git_sha()
        version_suffix = f".dev0+{git_sha}"
    elif release_type == "nightly":
        # Construct a nightly (pre-release) version:
        # https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases
        current_date = get_current_date()
        version_suffix = f"rc{current_date}"
    else:
        raise ValueError(f"Unhandled release type '{release_type}'")
    _log(f"Version suffix: '{version_suffix}'")

    rocm_package_version = base_version + version_suffix
    _log(f"Full version  : '{rocm_package_version}'")

    return rocm_package_version


def main(argv):
    parser = argparse.ArgumentParser(prog="compute_rocm_package_version")

    release_type_group = parser.add_mutually_exclusive_group()
    release_type_group.add_argument(
        "--release-type",
        type=str,
        choices=["dev", "nightly"],
        help="The type of package version to produce",
    )
    release_type_group.add_argument(
        "--custom-version-suffix",
        type=str,
        help="Custom version suffix to use instead of an automatic suffix",
    )

    parser.add_argument(
        "--override-base-version",
        type=str,
        help="Override the base version from version.json with this value",
    )

    args = parser.parse_args(argv)

    rocm_package_version = compute_version(
        args.release_type,
        args.custom_version_suffix,
        args.override_base_version,
    )
    gha_set_output({"rocm_package_version": rocm_package_version})


if __name__ == "__main__":
    main(sys.argv[1:])
