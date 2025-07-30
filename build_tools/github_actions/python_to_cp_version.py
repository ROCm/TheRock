#!/usr/bin/env python
"""Determines part of the path to the Python interpreter in a manylinux image.

Example usage:

    python python_to_cp_version.py --python-version 3.12

  The following string is appended to the file specified in the "GITHUB_ENV"
  environment variable:

    cp_version=cp312-cp312

Writing the output to the "GITHUB_ENV" file can be suppressed by passing
`--no-write-env-file`.
"""

import argparse
import re
import sys

from github_actions_utils import *


def is_version(version) -> bool:
    return re.match(r"^([1-9])\.((0|([1-9][0-9]*)))(t)?$", version) is not None


def transform_python_version(python_version: str) -> str:
    if not is_version(python_version):
        raise ValueError(f"Version '{python_version}' did not match accepted regex")

    version_without_dot = python_version.replace(".", "")
    cp_version = f"cp{version_without_dot}-cp{version_without_dot}"
    return cp_version


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="python_to_cp_version.py")
    p.add_argument(
        "--python-version",
        required=True,
        type=str,
        help="Python version to be transformed (e.g. 3.12)",
    )
    p.add_argument(
        "--write-env-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write `cp_version` to GITHUB_ENV file",
    )
    args = p.parse_args(argv)

    cp_version = transform_python_version(args.python_version)
    print(cp_version)

    if args.write_env_file:
        gha_set_env({"cp_version": cp_version})


if __name__ == "__main__":
    main(sys.argv[1:])
