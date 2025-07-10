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
import os
import re
import sys


def is_version(version) -> bool:
    return re.match(r"^([1-9])\.((0|([1-9][0-9]*)))(t)?$", version) is not None


def transform_python_version(python_version: str) -> str:
    if not is_version(python_version):
        raise Exception("Invalid version")

    tmp = python_version.replace(".", "")
    cp_version = f"cp{tmp}-cp{tmp}"
    return cp_version


def write_env_file(cp_version: str):
    env_file = os.getenv("GITHUB_ENV")

    with open(env_file, "a") as f:
        f.write(f"cp_version={cp_version}")


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="python_to_cp_version.py")
    p.add_argument(
        "--python-version",
        required=True,
        type=str,
        help="Python version to be transformed",
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
        write_env_file(cp_version)


if __name__ == "__main__":
    main(sys.argv[1:])
