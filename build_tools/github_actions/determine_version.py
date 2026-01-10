#!/usr/bin/env python
"""Determines the SDK version and version suffix to pass as additional
arguments to scripts like `external-builds/pytorch/build_prod_wheels.py`.

Example usage:

    python determine_version.py --rocm-version 7.0.0 --write-env-file

  The following string is appended to the file specified in the "GITHUB_ENV"
  environment variable:

    optional_build_prod_arguments=--rocm-sdk-version ==7.0.0 --version-suffix +rocm7.0.0

Writing the output to the "GITHUB_ENV" file can be suppressed by passing
`--no-write-env-file`.
"""

# TODO: Rename to something more like "forward rocm to version to pytorch build"?
#       This overlaps with compute_package_version used for rocm python packages
#       Maybe unify with write_torch_versions.py too? These should also work
#       together and follow a similar style.
# TODO: Or share with JAX builds? Could use a different name in that case too.

from packaging.version import parse

import argparse
import sys

from github_actions_utils import *


def derive_version_suffix(rocm_version: str) -> str:
    # Compute a version suffix to be used as a local version identifier:
    # https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers
    #
    # For example, torch with base version `2.9.0` built with rocm `7.10.0`
    # support can use this suffix in its version as `2.9.0+rocm7.10.0`.
    #
    # We take extra care here to sort final > nightly > dev and only include
    # a single `+` in the suffix.
    #
    # For example:
    #
    # | description | rocm version       | suffix                     |
    # | ----------- | ------------------ | -------------------------- |
    # | final       | 7.10.0             | +rocm7.10.0                |
    # | nightly     | 7.10.0a20251124    | +rocm7.10.0a20251124       |
    # | dev         | 7.10.0.dev0+efed3c | +devrocm7.10.0.dev0-efed3c |
    #                                       ^                 ^
    #                                       |                 |-- no `+` here
    #                                       |
    #                                       |--- devrocm sorts older than rocm

    parsed_version = parse(rocm_version)
    base_name = "devrocm" if "dev" in rocm_version else "rocm"
    version_suffix = f"+{base_name}{str(parsed_version).replace('+','-')}"

    return version_suffix


def derive_versions(rocm_version: str, verbose_output: bool) -> str:
    parsed_version = parse(rocm_version)
    rocm_sdk_version = f"=={parsed_version}"

    version_suffix = derive_version_suffix(rocm_version)

    optional_build_prod_arguments = (
        f"--rocm-sdk-version {rocm_sdk_version} --version-suffix {version_suffix}"
    )

    if verbose_output:
        print(f"ROCm version: {parsed_version}")
        print(f"`--rocm-sdk-version`\t: {rocm_sdk_version}")
        print(f"`--version-suffix`\t: {version_suffix}")
        print()

    return optional_build_prod_arguments


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="determine_version.py")
    p.add_argument(
        "--rocm-version",
        required=True,
        type=str,
        help="ROCm version to derive the parameters from",
    )
    p.add_argument(
        "--write-env-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write `optional_build_prod_arguments` to GITHUB_ENV file",
    )
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verbose output",
    )
    args = p.parse_args(argv)

    optional_build_prod_arguments = derive_versions(args.rocm_version, args.verbose)
    print(f"{optional_build_prod_arguments}")

    if args.write_env_file:
        gha_set_env({"optional_build_prod_arguments": optional_build_prod_arguments})


if __name__ == "__main__":
    main(sys.argv[1:])
