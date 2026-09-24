# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# CK's two-architecture device translation units repeatedly crash AMD Clang's
# heterogeneous-DWARF emission when the top-level RelWithDebInfo flags leak
# into its otherwise Release-only build. Retain TSAN for its host code
# and remove only debug-info generation from this device-heavy subproject.
include("${THEROCK_SOURCE_DIR}/cmake/therock_tsan_device_build.cmake")
therock_tsan_strip_debug_info()
