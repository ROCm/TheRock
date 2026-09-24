# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# hipTensor instantiates the same large, two-architecture CK contraction
# templates that trigger AMD Clang's heterogeneous-DWARF crash. Retain host
# TSAN instrumentation and the full device build while omitting debug info.
include("${THEROCK_SOURCE_DIR}/cmake/therock_tsan_device_build.cmake")
therock_tsan_strip_debug_info()
