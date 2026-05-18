# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch build matrix for workflows."""

import json
import os

from github_actions_api import gha_set_output

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

PYTORCH_REFS_LINUX = [
    "release/2.8",
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "release/2.12",
    "nightly",
]

PYTORCH_REFS_WINDOWS = [
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "release/2.12",
    "nightly",
]


def generate_pytorch_matrix(
    python_version: str | None, amdgpu_family: str, platform: str = "linux"
) -> list[dict]:
    versions = [python_version] if python_version else PYTHON_VERSIONS
    pytorch_refs = PYTORCH_REFS_WINDOWS if platform == "windows" else PYTORCH_REFS_LINUX
    matrix = []

    for py in versions:
        for ref in pytorch_refs:
            # Python 3.14 support added in PyTorch 2.9
            if py == "3.14" and ref == "release/2.8":
                continue
            # gfx1153 support added in PyTorch 2.10
            if amdgpu_family == "gfx1153" and ref in ["release/2.8", "release/2.9"]:
                continue
            matrix.append({"python_version": py, "pytorch_git_ref": ref})

    return matrix


def main():
    python_version = os.getenv("PYTHON_VERSION") or None
    amdgpu_family = os.getenv("AMDGPU_FAMILY", "")
    platform = os.getenv("PLATFORM", "linux")
    matrix = generate_pytorch_matrix(python_version, amdgpu_family, platform)
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})


if __name__ == "__main__":
    main()
