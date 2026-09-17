# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# CK's two-architecture device translation units repeatedly crash AMD Clang's
# heterogeneous-DWARF emission when the top-level RelWithDebInfo flags leak
# into its otherwise Release-only build. CK is a build dependency rather than
# an admitted phase-one test component, so retain HOST_TSAN for its host code
# but remove only debug-info generation from this device-heavy subproject.
include("${THEROCK_SOURCE_DIR}/cmake/therock_host_tsan_device_build.cmake")
therock_host_tsan_strip_debug_info()
