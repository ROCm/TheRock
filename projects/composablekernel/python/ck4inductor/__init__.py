# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


def __version__():
    import subprocess

    # needs to be manually updated
    rocm_version = "7.1.1"
    hash_width = 6
    try:
        hash = subprocess.check_output("git rev-parse HEAD", shell=True, text=True)[
            :hash_width
        ]
    except Exception:
        hash = "0" * hash_width
    try:
        change_count = subprocess.check_output(
            f"git rev-list rocm-{rocm_version}..HEAD --count", shell=True, text=True
        ).strip()
    except Exception:
        change_count = "0"
    return f"{rocm_version}.dev{change_count}+g{hash}"
