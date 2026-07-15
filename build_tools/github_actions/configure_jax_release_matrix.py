#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate JAX release build matrix for workflows."""

import argparse
import json
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
JAX_REFS = [
    {
        "jax_ref": "rocm-jaxlib-v0.9.1",
        "jax_repository": "ROCm/rocm-jax",
        "build_mode": "native",
        "gfx_arch": "",
    },
    {
        "jax_ref": "rocm-jaxlib-v0.10.0",
        "jax_repository": "ROCm/jax",
        "build_mode": "manylinux",
        "gfx_arch": "device-all",
    },
    {
        "jax_ref": "rocm-jaxlib-v0.10.2",
        "jax_repository": "ROCm/jax",
        "build_mode": "manylinux",
        "gfx_arch": "device-all",
    },
]


def _split_values(raw: str) -> list[str]:
    """Split comma, semicolon, or whitespace-separated workflow input values."""
    return [
        value.strip()
        for value in raw.replace(",", " ").replace(";", " ").split()
        if value.strip()
    ]


def generate_jax_matrix(
    python_versions: list[str] | None,
) -> list[dict[str, object]]:
    versions = python_versions if python_versions else PYTHON_VERSIONS
    matrix: list[dict[str, object]] = []
    for py in versions:
        for ref_cfg in JAX_REFS:
            # These row keys are the contract with workflow files which use them
            # via matrix.<key> expressions. Empty values are allowed when the
            # workflow handles them explicitly, but undefined keys are not.
            matrix.append(
                {
                    "python_version": py,
                    "jax_ref": ref_cfg["jax_ref"],
                    "jax_repository": ref_cfg["jax_repository"],
                    "build_mode": ref_cfg["build_mode"],
                    # gfx_arch is intentionally empty for native JAX builds and
                    # non-empty for manylinux builds. This direct lookup raises
                    # KeyError if JAX_REFS omits the key.
                    "gfx_arch": ref_cfg["gfx_arch"],
                }
            )
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate JAX release build matrix")
    parser.add_argument(
        "--python-versions",
        type=str,
        default="",
        help="Comma, semicolon, or whitespace separated list of Python versions (default: all)",
    )
    args = parser.parse_args(argv)

    python_versions = _split_values(args.python_versions) or None

    matrix = generate_jax_matrix(python_versions)
    gha_set_output({"jax_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
