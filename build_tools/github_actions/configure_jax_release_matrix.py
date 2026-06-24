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
        "build_jaxlib": False,
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
            matrix.append(
                {
                    "python_version": py,
                    "jax_ref": ref_cfg["jax_ref"],
                    "build_jaxlib": ref_cfg["build_jaxlib"],
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
