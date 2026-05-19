#!/usr/bin/env python

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Computes a ROCm package version with an appropriate suffix for a given release type.

For usage from other Python scripts, call the `compute_version()` function
directly. When used from GitHub Actions, this writes all three version outputs
to GITHUB_OUTPUT:
  - 'rocm_package_version' for wheel packages
  - 'rocm_deb_package_version' for deb packages
  - 'rocm_rpm_package_version' for rpm packages

Sample usage:

  python compute_rocm_package_version.py --release-type=dev
  # rocm_package_version=7.10.0.dev0+f689a8ea40232f3f6be1ec958354b108349023ff
  # rocm_deb_package_version=7.10.0~dev20251203
  # rocm_rpm_package_version=7.10.0~20251203gf689a8e

  python compute_rocm_package_version.py --release-type=prerelease --prerelease-version=2
  # rocm_package_version=7.10.0rc2
  # rocm_deb_package_version=7.10.0~pre2
  # rocm_rpm_package_version=7.10.0~rc2

  python compute_rocm_package_version.py --release-type=nightly
  # rocm_package_version=7.10.0a20251021
  # rocm_deb_package_version=7.10.0~20251021
  # rocm_rpm_package_version=7.10.0~20251021

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

from github_actions.github_actions_api import *

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


def get_git_sha(short: bool = False, override_git_sha: str | None = None):
    """Gets the git SHA to embed in version strings.

    Resolution order:
      1. ``override_git_sha`` (explicit caller override)
      2. ``GITHUB_SHA`` environment variable
      3. ``git rev-parse HEAD`` in the repo checkout

    When the workflow is triggered cross-repo (e.g. rockrel calling TheRock),
    GITHUB_SHA refers to the *caller's* commit.  Pass ``override_git_sha``
    from the checkout step to get the correct TheRock SHA.
    """

    if override_git_sha:
        git_sha = override_git_sha
    else:
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

    # Shorten the sha to 8 characters if requested
    if short:
        git_sha = git_sha[:8]

    return git_sha


def get_current_date():
    """Gets the current date as YYYYMMDD."""
    return datetime.today().strftime("%Y%m%d")


def compute_version(
    package_type: str = "wheel",
    release_type: str | None = None,
    custom_version_suffix: str | None = None,
    prerelease_version: str | None = None,
    override_base_version: str | None = None,
    override_git_sha: str | None = None,
) -> str:
    """Compute package version based on package type and release type.

    Args:
        package_type: Type of package ("wheel", "deb", or "rpm")
        release_type: Release type ("dev", "nightly", "prerelease", or "release")
        custom_version_suffix: Custom suffix to override automatic suffix
        prerelease_version: Prerelease version number
        override_base_version: Override the base version from version.json
        override_git_sha: Explicit git SHA override, forwarded to get_git_sha().
            See get_git_sha() for details on when this is needed.

    Returns:
        Computed version string appropriate for the package type
    """
    if override_base_version:
        base_version = override_base_version
    else:
        base_version = load_rocm_version()
    _log(f"Base version  : '{base_version}'")
    _log(f"Package type  : '{package_type}'")

    # Handle wheel packages (Python packaging standards)
    if package_type == "wheel":
        if custom_version_suffix:
            # Trust the custom suffix to satisfy the general rules:
            # https://packaging.python.org/en/latest/specifications/version-specifiers/
            version_suffix = custom_version_suffix
        elif release_type == "dev":
            # Construct a dev release version:
            # https://packaging.python.org/en/latest/specifications/version-specifiers/#developmental-releases
            git_sha = get_git_sha(override_git_sha=override_git_sha)
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
            raise ValueError(
                f"Unhandled release type '{release_type}' for wheel packages"
            )
        _log(f"Version suffix: '{version_suffix}'")

        rocm_package_version = base_version + version_suffix
        _log(f"Full version  : '{rocm_package_version}'")

        return rocm_package_version

    # Handle native packages (deb/rpm)
    else:  # package_type in ["deb", "rpm"]
        if custom_version_suffix:
            # Custom suffix uses the provided value
            version_suffix_str = f"{custom_version_suffix}"
        elif release_type == "release":
            # Final release version - no suffix
            # Format: <rocm-version>
            version_suffix_str = ""
        elif release_type == "dev":
            # Construct a dev release version with date and optionally git SHA
            # deb format: <rocm-version>~dev<YYYYMMDD>
            # rpm format: <rocm-version>~<YYYYMMDD>g<short-git-sha>
            current_date = get_current_date()
            if package_type == "deb":
                version_suffix_str = f"~dev{current_date}"
            else:  # rpm
                git_sha = get_git_sha(short=True, override_git_sha=override_git_sha)
                version_suffix_str = f"~{current_date}g{git_sha}"
        elif release_type == "nightly":
            # Construct a nightly version with date
            # Format: <rocm-version>~<YYYYMMDD>
            current_date = get_current_date()
            version_suffix_str = f"~{current_date}"
        elif release_type == "prerelease":
            # Construct a prerelease version
            # deb format: <rocm-version>~pre<N>
            # rpm format: <rocm-version>~rc<N>
            if package_type == "deb":
                version_suffix_str = f"~pre{prerelease_version}"
            else:  # rpm
                version_suffix_str = f"~rc{prerelease_version}"
        else:
            raise ValueError(
                f"Unhandled release type '{release_type}' for {package_type} packages"
            )
        _log(f"Version suffix: '{version_suffix_str}'")

        rocm_package_version = base_version + version_suffix_str
        _log(f"Full version  : '{rocm_package_version}'")

        return rocm_package_version


def main(argv):
    parser = argparse.ArgumentParser(prog="compute_rocm_package_version")

    release_type_group = parser.add_mutually_exclusive_group()
    release_type_group.add_argument(
        "--release-type",
        type=str,
        choices=["dev", "nightly", "prerelease", "release"],
        help="The type of package version to produce (note: 'release' only valid for deb/rpm)",
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

    parser.add_argument(
        "--override-git-sha",
        type=str,
        help="Explicit git SHA to embed in the version instead of auto-detecting",
    )

    args = parser.parse_args(argv)

    # Validation
    if args.release_type == "release":
        parser.error("'release' type is only valid for deb/rpm packages, not wheel")

    if args.release_type != "prerelease" and args.prerelease_version:
        parser.error("release type must be 'prerelease' if --prerelease-version is set")
    elif args.release_type == "prerelease" and not args.prerelease_version:
        parser.error(
            "--prerelease-version is required when release type is 'prerelease'"
        )

    # Compute versions for all three package types: wheel, deb, and rpm
    outputs = {}
    for pkg_type in ["wheel", "deb", "rpm"]:
        version = compute_version(
            pkg_type,
            args.release_type,
            args.custom_version_suffix,
            args.prerelease_version,
            args.override_base_version,
        )

        # Set appropriate output variable based on package type
        if pkg_type == "wheel":
            outputs["rocm_package_version"] = version
        else:  # deb or rpm
            outputs[f"rocm_{pkg_type}_package_version"] = version

    gha_set_output(outputs)


if __name__ == "__main__":
    main(sys.argv[1:])
