# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# hipTensor instantiates the same large, two-architecture CK contraction
# templates that trigger AMD Clang's heterogeneous-DWARF crash. hipTensor is
# outside the phase-one test matrix; retain host TSAN instrumentation and the
# full device build while omitting only debug-info generation.
include("${THEROCK_SOURCE_DIR}/cmake/therock_host_tsan_device_build.cmake")
therock_host_tsan_strip_debug_info()
