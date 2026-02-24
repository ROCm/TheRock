#!/usr/bin/env python
"""Computes JAX package version from rocm_version and JAX requirements.

Used as a fallback when the test workflow is triggered via manual
workflow_dispatch (without versions passed from a build workflow).

The version suffix is computed to match what rocm-jax's ci_build produces:
    +rocm{rocm_version.replace('+', '.')}

This differs from PyTorch's derive_version_suffix() which uses a +devrocm
prefix for dev builds. JAX always uses +rocm and replaces '+' with '.' in
the local version segment for PEP 440 compliance.

Example usage:

    python compute_jax_package_version.py \
        --rocm-version 7.12.0.dev0+e1a5d395 \
        --jax-requirements jax/build/requirements.txt

  The following strings are appended to the file specified in the "GITHUB_ENV"
  environment variable:

    JAX_VERSION=0.8.0
    JAXLIB_VERSION=0.8.0+rocm7.12.0.dev0.e1a5d395
"""

import argparse
import re
import sys

from github_actions_utils import *


def extract_jax_version_from_requirements(requirements_path: str) -> str:
    """Extracts the JAX version from a requirements.txt file.

    Looks for lines like 'jax==0.8.0' or 'jaxlib==0.8.0' and returns
    the version number.
    """
    pattern = re.compile(r"^\s*(jax|jaxlib)\s*==\s*([^#\s]+)")

    with open(requirements_path, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                return match.group(2)

    raise ValueError(f"Could not find jax or jaxlib version in '{requirements_path}'")


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="compute_jax_package_version.py")
    p.add_argument(
        "--rocm-version",
        required=True,
        type=str,
        help="ROCm version (e.g. 7.12.0, 7.12.0.dev0+hash)",
    )
    p.add_argument(
        "--jax-requirements",
        required=True,
        type=str,
        help="Path to JAX requirements.txt file",
    )

    args = p.parse_args(argv)

    jax_version = extract_jax_version_from_requirements(args.jax_requirements)

    # Match the version suffix format that rocm-jax ci_build produces:
    #   +rocm{rocm_version.replace('+', '.')}
    # This replaces '+' with '.' for PEP 440 compliance (only one '+' allowed).
    sanitized_rocm = args.rocm_version.replace("+", ".")
    jaxlib_version = f"{jax_version}+rocm{sanitized_rocm}"

    print(f"JAX_VERSION={jax_version}")
    print(f"JAXLIB_VERSION={jaxlib_version}")

    gha_set_env(
        {
            "JAX_VERSION": jax_version,
            "JAXLIB_VERSION": jaxlib_version,
        }
    )


if __name__ == "__main__":
    main(sys.argv[1:])
