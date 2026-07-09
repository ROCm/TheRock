#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate JAX build matrices for CI and release workflows."""

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

CI_JAX_REFS = {"rocm-jaxlib-v0.10.0", "rocm-jaxlib-v0.10.2"}
CI_PYTHON_VERSIONS = PYTHON_VERSIONS


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
                    "jax_repository": ref_cfg["jax_repository"],
                    "build_mode": ref_cfg["build_mode"],
                    "gfx_arch": ref_cfg["gfx_arch"],
                }
            )
    return matrix


def generate_jax_matrix_for_ci(
    python_versions: list[str] | None,
) -> list[dict[str, str]]:
    """Generate the JAX matrix used by Multi-Arch CI.

    The release matrix includes legacy/native JAX entries such as
    rocm-jaxlib-v0.9.1. The CI workflow is manylinux-only, so filter the
    release matrix to the supported manylinux entries.
    """
    versions = python_versions if python_versions else CI_PYTHON_VERSIONS

    matrix: list[dict[str, str]] = []
    for cell in generate_jax_matrix(versions):
        build_mode = str(cell["build_mode"])
        jax_ref = str(cell["jax_ref"])

        if build_mode != "manylinux":
            continue
        if jax_ref not in CI_JAX_REFS:
            continue

        matrix.append(
            {
                "python_version": str(cell["python_version"]),
                "jax_ref": jax_ref,
                "jax_repository": str(cell["jax_repository"]),
                "build_mode": build_mode,
                "gfx_arch": str(cell["gfx_arch"]),
            }
        )

    if not matrix:
        raise ValueError(
            "No supported manylinux JAX CI matrix entries were generated. "
            f"Allowed refs: {sorted(CI_JAX_REFS)}"
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
