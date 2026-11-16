# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate Rocjitsu's native files in the matching development SDK."""

from rocm_sdk._devel import get_devel_root
from rocm_sdk_devel._dist_info import THIS_PACKAGE_ENTRY

# Initialize the SDK's links before the launcher looks up its native files.
get_devel_root()
