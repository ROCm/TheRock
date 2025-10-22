#!/usr/bin/env python

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Computes a ROCm package version with an appropriate suffix for a given release type.

For usage from other Python scripts, call the `compute_version()` function
directly. WHen used from GitHub Actions, this writes to 'version' in GITHUB_OUTPUT.

Sample usage with standard release versions:

  python compute_rocm_package_version.py --release-type=dev
  # 7.10.0.dev0+f689a8ea40232f3f6be1ec958354b108349023ff

  python compute_rocm_package_version.py --release-type=prerelease --prerelease-version=2
  # 7.10.0rc2

  python compute_rocm_package_version.py --release-type=nightly
  # 7.10.0a20251021

Sample usage with custom release versions:

  python compute_rocm_package_version.py --custom-version-suffix=.dev0
  # 7.10.0.dev0

  python compute_rocm_package_version.py --release-type=nightly --override-base-version=7.99.0
  # 7.99.0a20251021
"""

import argparse
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

from github_actions.github_actions_utils import *

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent


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

    # Default GitHub environment variable, info:
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
    release_type: str | None = None,
    custom_version_suffix: str | None = None,
    prerelease_version: str | None = None,
    override_base_version: str | None = None,
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
        # Construct a nightly (a / "alpha") version:
        # https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases
        current_date = get_current_date()
        version_suffix = f"a{current_date}"
    elif release_type == "prerelease":
        # Construct a prerelease (rc / "release candidate") version
        # https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases
        version_suffix = f"rc{prerelease_version}"
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
        choices=["dev", "nightly", "prerelease"],
        help="The type of package version to produce",
    )
    release_type_group.add_argument(
        "--custom-version-suffix",
        type=str,
        help="Custom version suffix to use instead of an automatic suffix",
    )

    parser.add_argument(
        "--prerelease-version",
        type=str,
        help="Prerelease version (typically a build number)",
    )

    parser.add_argument(
        "--override-base-version",
        type=str,
        help="Override the base version from version.json with this value",
    )

    args = parser.parse_args(argv)

    if args.release_type != "prerelease" and args.prerelease_version:
        parser.error("release type must be 'prerelease' if --prerelease-version is set")
    elif args.release_type == "prerelease" and not args.prerelease_version:
        parser.error(
            "--prerelease-version is required when release type is 'prerelease'"
        )

    rocm_package_version = compute_version(
        args.release_type,
        args.custom_version_suffix,
        args.prerelease_version,
        args.override_base_version,
    )
    gha_set_output({"rocm_package_version": rocm_package_version})


if __name__ == "__main__":
    main(sys.argv[1:])
