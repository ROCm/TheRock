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

RELEASE_TYPES = ["ci", "dev", "nightly", "prerelease"]

# TODO: add opt-ins for CI runs to use python versions and JAX refs normally
#       only included in release runs.
RELEASE_PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
CI_PYTHON_VERSIONS = {
    "linux": ["3.12"],
}

JAX_REF_CONFIGS = {
    "rocm-jaxlib-v0.10.0": {
        "jax_ref": "rocm-jaxlib-v0.10.0",
        "jax_repository": "ROCm/jax",
        "gfx_arch": "device-all",
    },
    "rocm-jaxlib-v0.10.1": {
        "jax_ref": "rocm-jaxlib-v0.10.1",
        "jax_repository": "ROCm/jax",
        "gfx_arch": "device-all",
    },
    "rocm-jaxlib-v0.10.2": {
        "jax_ref": "rocm-jaxlib-v0.10.2",
        "jax_repository": "ROCm/jax",
        "gfx_arch": "device-all",
    },
    "rocm-jaxlib-v0.11.0": {
        "jax_ref": "rocm-jaxlib-v0.11.0",
        "jax_repository": "ROCm/jax",
        "gfx_arch": "device-all",
        # JAX dropped Python 3.11 support in 0.11.0.
        "exclude_python_versions": ["3.11"],
    },
}

# Keep release behavior equivalent to the old generate_jax_matrix(None):
# all release refs across all release Python versions.
#
# TODO: separate out nightly/dev/prerelease JAX refs if those release types
# should differ later.
RELEASE_JAX_REFS = {
    "linux": [
        "rocm-jaxlib-v0.10.0",
        "rocm-jaxlib-v0.10.1",
        "rocm-jaxlib-v0.10.2",
        "rocm-jaxlib-v0.11.0",
    ],
}

# CI builds a single, stable JAX ref to keep the CI runner load low; the base
# CI configuration favors breadth across GPU targets/platforms/frameworks
# rather than every release version. Additional refs can be opted in as needed.
CI_JAX_REFS = {
    "linux": [
        "rocm-jaxlib-v0.11.0",
    ],
}


def _split_values(raw: str) -> list[str]:
    """Split comma, semicolon, or whitespace-separated workflow input values."""
    return [
        value.strip()
        for value in raw.replace(",", " ").replace(";", " ").split()
        if value.strip()
    ]


def _default_python_versions(*, release_type: str, platform: str) -> list[str]:
    if release_type == "ci":
        return list(CI_PYTHON_VERSIONS[platform])
    return list(RELEASE_PYTHON_VERSIONS)


def _default_jax_refs(*, release_type: str, platform: str) -> list[str]:
    if release_type == "ci":
        return list(CI_JAX_REFS[platform])
    return list(RELEASE_JAX_REFS[platform])


def generate_jax_matrix(
    *,
    jax_refs: list[str],
    python_versions: list[str],
) -> list[dict[str, str]]:
    matrix: list[dict[str, str]] = []
    for py in python_versions:
        for ref in jax_refs:
            ref_cfg = JAX_REF_CONFIGS[ref]
            # Skip Python versions a ref explicitly excludes (declared in
            # JAX_REF_CONFIGS, e.g. JAX 0.11.0 dropped Python 3.11).
            if py in ref_cfg.get("exclude_python_versions", ()):
                continue
            # These row keys are the contract with workflow files which use them
            # via matrix.<key> expressions. Empty values are allowed when the
            # workflow handles them explicitly, but undefined keys are not.
            matrix.append(
                {
                    "python_version": py,
                    "jax_ref": ref_cfg["jax_ref"],
                    "jax_repository": ref_cfg["jax_repository"],
                    # gfx_arch selects the ROCm device package for the manylinux
                    # build (e.g. device-all). This direct lookup raises
                    # KeyError if JAX_REF_CONFIGS omits the key.
                    "gfx_arch": ref_cfg["gfx_arch"],
                }
            )

    return matrix


def generate_jax_matrix_for_release_type(
    *,
    release_type: str,
    platform: str,
    python_versions: list[str] | None = None,
    jax_refs: list[str] | None = None,
) -> list[dict[str, str]]:
    if release_type not in RELEASE_TYPES:
        raise ValueError(f"Unknown release_type: {release_type!r}")
    if platform not in ["linux"]:
        raise ValueError(f"Unknown platform: {platform!r}")

    versions = python_versions or _default_python_versions(
        release_type=release_type,
        platform=platform,
    )
    refs = jax_refs or _default_jax_refs(
        release_type=release_type,
        platform=platform,
    )

    unknown_refs = sorted(set(refs) - set(JAX_REF_CONFIGS))
    if unknown_refs:
        raise ValueError(f"Unknown JAX refs: {unknown_refs!r}")

    return generate_jax_matrix(
        jax_refs=refs,
        python_versions=versions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate JAX build matrix")
    parser.add_argument(
        "--python-versions",
        type=str,
        default="",
        help=(
            "Comma, semicolon, or whitespace separated list of Python versions "
            "(default depends on --release-type)"
        ),
    )
    parser.add_argument(
        "--jax-refs",
        type=str,
        default="",
        help=(
            "Comma, semicolon, or whitespace separated list of JAX refs "
            "(default depends on --release-type and --platform)"
        ),
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="linux",
        choices=["linux"],
        help="Platform to generate matrix for (default: linux)",
    )
    parser.add_argument(
        "--release-type",
        type=str,
        default="dev",
        choices=RELEASE_TYPES,
        help="Release type selecting default JAX/Python matrix (default: dev)",
    )
    args = parser.parse_args(argv)

    python_versions = _split_values(args.python_versions) or None
    jax_refs = _split_values(args.jax_refs) or None

    matrix = generate_jax_matrix_for_release_type(
        release_type=args.release_type,
        platform=args.platform,
        python_versions=python_versions,
        jax_refs=jax_refs,
    )
    gha_set_output({"jax_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
