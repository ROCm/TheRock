# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Workaround: Disable ROCMChecks variable_watch before it loads, preventing
# ASAN build failures from Google Benchmark modifying CMAKE_CXX_FLAGS.
# Remove once upstream fix lands: https://github.com/ROCm/rocm-libraries/pull/6338
set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "" FORCE)
