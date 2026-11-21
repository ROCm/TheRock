#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Picks the version suffix and the wheels for one JAX wheel build.

Everything comes from the JAX checkout and the ROCm version, never from a
package index, so the same commit built against the same ROCm always gets the
same versions.

`--wheel-type release` is for refs whose jax and jaxlib are on PyPI. Only the
ROCm plugin and PJRT wheels are built, stamped with the ROCm suffix alone:

    0.11.2+rocm7.14.0a20261002

`--wheel-type nightly` is for refs with no PyPI release, such as
jax-ml/jax@main. jax and jaxlib are built from the same checkout as the plugin,
and all four wheels carry a `.dev` date taken from the checkout's HEAD commit,
which keeps them PEP 440 pre-releases that `pip install` without `--pre` skips:

    0.12.0.dev20261002+rocm7.14.0a20261002

The date rides in ML_WHEEL_VERSION_SUFFIX with ML_WHEEL_TYPE left at release:
XLA's own "nightly" type drops the suffix, and its "custom" type spells the
local label differently from build/build.py's copy glob.

Example:

    python resolve_jax_version.py --wheel-type nightly \\
        --rocm-version 7.14.0a20261002 --jax-source-dir jax-source

writes these step outputs to GITHUB_OUTPUT:

    ml_wheel_version_suffix=.dev20261002+rocm7.14.0a20261002
    wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt
    bazel_options=

An ASAN `--build-variant` fills `bazel_options` with host-only ASAN flags. The
version needs no marker of its own: the ASAN ROCm version carries one, e.g.
`+rocm10.3.0a20261119-asandebug`, as for PyTorch.
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.configure_jax_release_matrix import (
    ASAN_BUILD_VARIANTS,
    BUILD_VARIANTS,
)
from github_actions.determine_version import derive_version_suffix
from github_actions.github_actions_api import gha_set_output

WHEELS = {
    "release": ("jax-rocm-plugin", "jax-rocm-pjrt"),
    "nightly": ("jax", "jaxlib", "jax-rocm-plugin", "jax-rocm-pjrt"),
}

# The ROCm toolchain pinned by the JAX release branches has no asan feature, so
# --config=asan_shared alone does not instrument. GPU kernels stay uninstrumented.
ASAN_BAZEL_OPTIONS = (
    "--bazel_options=--config=asan_shared"
    " --bazel_options=--copt=-fsanitize=address"
    " --bazel_options=--copt=-fno-gpu-sanitize"
    " --bazel_options=--linkopt=-fsanitize=address"
)


def commit_date(jax_source_dir: Path) -> str:
    """Returns the UTC committer date of the checkout's HEAD as YYYYMMDD."""
    timestamp = subprocess.run(
        ["git", "show", "-s", "--format=%ct", "HEAD"],
        cwd=jax_source_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return datetime.datetime.fromtimestamp(
        int(timestamp), tz=datetime.timezone.utc
    ).strftime("%Y%m%d")


def ml_wheel_version_suffix(
    wheel_type: str, rocm_version: str, jax_source_dir: Path
) -> str:
    """Returns the ML_WHEEL_VERSION_SUFFIX for this wheel type."""
    rocm_suffix = derive_version_suffix(rocm_version)
    if wheel_type == "nightly":
        return f".dev{commit_date(jax_source_dir)}{rocm_suffix}"
    return rocm_suffix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pick the version suffix and wheels for a JAX wheel build"
    )
    parser.add_argument(
        "--wheel-type",
        choices=sorted(WHEELS),
        required=True,
        help="release for refs on PyPI; nightly for refs with no release (main)",
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version the wheels are built against",
    )
    parser.add_argument(
        "--jax-source-dir",
        type=Path,
        required=True,
        help="JAX checkout being built",
    )
    parser.add_argument(
        "--build-variant",
        default="release",
        choices=BUILD_VARIANTS,
        help="TheRock build variant of the release (default: release)",
    )
    args = parser.parse_args(argv)

    suffix = ml_wheel_version_suffix(
        args.wheel_type, args.rocm_version, args.jax_source_dir
    )
    gha_set_output(
        {
            "ml_wheel_version_suffix": suffix,
            "wheels": ",".join(WHEELS[args.wheel_type]),
            "bazel_options": (
                ASAN_BAZEL_OPTIONS if args.build_variant in ASAN_BUILD_VARIANTS else ""
            ),
        }
    )


if __name__ == "__main__":
    main()
