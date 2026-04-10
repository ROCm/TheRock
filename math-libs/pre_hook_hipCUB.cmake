# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Workaround: hipCUB ASAN build failures with BUILD_BENCHMARK=ON.
# 1) Disable ROCMChecks variable_watch before it loads.
# 2) Skip Google Benchmark try_run regex detection (fails under ASAN).
# Remove once upstream fix lands: https://github.com/ROCm/rocm-libraries/pull/6338
set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "" FORCE)
set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)
set(RUN_HAVE_STD_REGEX 1 CACHE STRING "" FORCE)
