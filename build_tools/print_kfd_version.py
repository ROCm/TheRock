#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Prints the KFD (Kernel Fusion Driver) IOCTL version reported by /dev/kfd.

The version is queried via AMDKFD_IOC_GET_VERSION, the same mechanism used
by rocdbgapi to validate driver compatibility before attaching to a process.
rocdbgapi requires KFD IOCTL version >= 1.13 and < 2.0.
"""

import fcntl
import os
import struct
import sys
from typing import Optional, Tuple

# AMDKFD_IOC_GET_VERSION = _IOR('K', 0x01, struct { u32 major; u32 minor; })
# _IOR: direction=0x80, size=8, type='K'=0x4b, nr=0x01 -> 0x80084b01
AMDKFD_IOC_GET_VERSION = 0x80084B01

KFD_DEVICE = "/dev/kfd"

# Supported range as enforced by rocdbgapi (os_driver_kfd.cpp)
KFD_VERSION_MIN = (1, 13)
KFD_VERSION_MAX = (2, 0)  # exclusive


def get_kfd_version() -> Tuple[int, int]:
    fd = os.open(KFD_DEVICE, os.O_RDWR)
    try:
        buf = bytearray(8)
        fcntl.ioctl(fd, AMDKFD_IOC_GET_VERSION, buf)
        major, minor = struct.unpack("II", buf)
        return major, minor
    finally:
        os.close(fd)


def version_supported(major: int, minor: int) -> bool:
    v = (major, minor)
    return KFD_VERSION_MIN <= v < KFD_VERSION_MAX


def main(argv: Optional[list] = None) -> int:
    if not os.path.exists(KFD_DEVICE):
        print(f"error: {KFD_DEVICE} not found — is the AMDGPU driver loaded?")
        return 1

    try:
        major, minor = get_kfd_version()
    except OSError as e:
        print(f"error: failed to query KFD version from {KFD_DEVICE}: {e}")
        return 1

    supported = version_supported(major, minor)
    status = "supported" if supported else "NOT supported"
    print(f"KFD IOCTL version: {major}.{minor} ({status})")
    print(
        f"Required range: >= {KFD_VERSION_MIN[0]}.{KFD_VERSION_MIN[1]}"
        f" and < {KFD_VERSION_MAX[0]}.{KFD_VERSION_MAX[1]}"
        f" (rocdbgapi requirement)"
    )

    return 0 if supported else 1


if __name__ == "__main__":
    sys.exit(main())
