#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests an installed rocshmem4py wheel on the local GPU.

Run it with the interpreter of the environment the wheel was installed into:

    .venv-rocshmem4py/bin/python build_tools/github_actions/rocshmem4py_smoke.py
"""

import rocshmem4py


def main() -> None:
    rocshmem4py.set_hip_device_from_env()
    unique_id = rocshmem4py.rocshmem_get_uniqueid()
    rocshmem4py.rocshmem_init_attr(0, 1, unique_id)
    try:
        buffer = rocshmem4py.rocshmem_create_buffer(4096)
        if buffer.ptr <= 0:
            raise RuntimeError("rocshmem4py returned an invalid GPU buffer")
        buffer.free()
    finally:
        rocshmem4py.hip_device_synchronize()
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()


if __name__ == "__main__":
    main()
