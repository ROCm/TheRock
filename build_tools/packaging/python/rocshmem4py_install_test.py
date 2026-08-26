#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke-test an installed rocshmem4py wheel and its packaged ROCm runtime."""

import argparse
from importlib.metadata import requires, version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-rocm-version", required=True)
    args = parser.parse_args()

    import _rocshmem4py  # noqa: F401
    import rocshmem4py

    wheel_version = version("rocshmem4py")
    assert rocshmem4py.__version__ == wheel_version
    assert rocshmem4py.__rocshmem_version__

    expected_rocm_version = args.expected_rocm_version
    assert version("rocm-sdk-core") == expected_rocm_version
    assert f"rocm-sdk-core=={expected_rocm_version}" in (requires("rocshmem4py") or [])

    rocshmem4py.set_hip_device_from_env()
    unique_id = rocshmem4py.rocshmem_get_uniqueid()
    rocshmem4py.rocshmem_init_attr(0, 1, unique_id)
    try:
        buffer = rocshmem4py.rocshmem_create_buffer(4096)
        assert buffer.ptr > 0
        buffer.free()
    finally:
        rocshmem4py.hip_device_synchronize()
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()


if __name__ == "__main__":
    main()
