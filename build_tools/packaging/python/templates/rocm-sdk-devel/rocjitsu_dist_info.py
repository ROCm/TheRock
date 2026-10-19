# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate Rocjitsu's native files in the matching development SDK."""

from rocm_sdk._devel import get_devel_root
from rocm_sdk_devel._dist_info import THIS_PACKAGE_ENTRY

# The launcher imports this module on first use. Expand the SDK through its
# normal locking and RECORD-update path before looking up the native package.
get_devel_root()
