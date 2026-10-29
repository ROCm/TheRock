#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

import argparse
import sys
from pathlib import Path

from packaging.version import parse

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_env


def rocm_version_from_torch_version(torch_version: str) -> str:
    """Recover the ROCm version encoded by ``derive_version_suffix``.

    Split the raw local version. Packaging turns the encoded ``+`` (``-``) into ``.``.
    """
    if "+" not in torch_version:
        raise ValueError(
            f"torch version {torch_version!r} has no local ROCm version suffix"
        )
    local = torch_version.split("+", 1)[1]
    encoded = None
    for marker in ("devrocm", "rocm"):
        if local.startswith(marker):
            encoded = local[len(marker) :]
            break
        token = f".{marker}"
        idx = local.find(token)
        if idx != -1:
            encoded = local[idx + len(token) :]
            break
    if not encoded:
        raise ValueError(
            f"torch version {torch_version!r} has no local ROCm version suffix"
        )
    # Inverse of str(parsed).replace("+", "-"): only the public/local split.
    if "-" in encoded:
        public, local_part = encoded.split("-", 1)
        encoded = f"{public}+{local_part}"
    return str(parse(encoded))


def derive_version_suffix(rocm_version: str) -> str:
    """Compute a version suffix to be used as a local version identifier.

    See https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers

    For example, torch with base version ``2.9.0`` built with rocm ``7.10.0``
    support uses this suffix in its version as ``2.9.0+rocm7.10.0``.

    We take extra care to sort final > nightly > dev and only include
    a single ``+`` in the suffix.

    | description | rocm version       | suffix                     |
    | ----------- | ------------------ | -------------------------- |
    | final       | 7.10.0             | +rocm7.10.0                |
    | nightly     | 7.10.0a20251124    | +rocm7.10.0a20251124       |
    | dev         | 7.10.0.dev0+efed3c | +devrocm7.10.0.dev0-efed3c |
                                            ^                 ^
                                            |                 |-- no ``+`` here
                                            |
                                            |--- devrocm sorts older than rocm
    """
    parsed = parse(rocm_version)
    base_name = "devrocm" if "dev" in rocm_version else "rocm"
    return f"+{base_name}{str(parsed).replace('+', '-')}"


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="determine_version.py")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--rocm-version",
        type=str,
        help="ROCm version to derive the parameters from",
    )
    source.add_argument(
        "--torch-version",
        type=str,
        help="Torch version whose local suffix encodes the ROCm version to print",
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

    if args.torch_version:
        # Print only the version. Callers pin `rocm[devel]==<version>` from this.
        print(rocm_version_from_torch_version(args.torch_version))
        return

    parsed_version = parse(args.rocm_version)
    version_suffix = derive_version_suffix(args.rocm_version)
    rocm_sdk_version = f"=={parsed_version}"
    optional_build_prod_arguments = (
        f"--rocm-sdk-version {rocm_sdk_version} --version-suffix {version_suffix}"
    )

    if args.verbose:
        print(f"ROCm version: {parsed_version}")
        print(f"`--rocm-sdk-version`\t: {rocm_sdk_version}")
        print(f"`--version-suffix`\t: {version_suffix}")
        print()

    print(f"{optional_build_prod_arguments}")

    if args.write_env_file:
        gha_set_env(
            {
                "optional_build_prod_arguments": optional_build_prod_arguments,
                "version_suffix": version_suffix,
            }
        )


if __name__ == "__main__":
    main(sys.argv[1:])
