#!/usr/bin/env python
"""Determines the SDK version and version suffix to pass as additional
arguments to `external-builds/pytorch/build_prod_wheels.py`.

Example usage:

    python determine_version.py --rocm-version 7.0.0 --write-env-file
"""

from packaging.version import Version, parse

import argparse
import os
import sys


def derive_versions(args):
    version = parse(args.rocm_version)
    rocm_sdk_version = f"=={version}"
    version_suffix = f"+rocm{str(version).replace('+','-')}"
    optional_build_prod_arguments = (
        f"--rocm-sdk-version {rocm_sdk_version} --version-suffix {version_suffix}"
    )

    if args.verbose:
        print(f"ROCm version: {version}")
        print(f"`--rocm-sdk-version`\t: {rocm_sdk_version}")
        print(f"`--version-suffix`\t: {version_suffix}")
        print()
    print(f"{optional_build_prod_arguments}")

    if args.write_env_file:
        env_file = os.getenv("GITHUB_ENV")

        with open(env_file, "a") as f:
            f.write(f"optional_build_prod_arguments={optional_build_prod_arguments}")


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

    derive_versions(args)


if __name__ == "__main__":
    main(sys.argv[1:])
