# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocBLAS has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()

# Enable BUILD_ADDRESS_SANITIZER when THEROCK_SANITIZER is ASAN.
# This enables RCCL's LTO optimization bypass for faster ASAN link times and
# turns on device-side ASAN instrumentation (xnack+).
#
# Caveat: RCCL hard-fails if BUILD_ADDRESS_SANITIZER is set together with
# more than one entry in GPU_TARGETS, because ASAN-instrumented device code
# combined with -fgpu-rdc exceeds the clang-offload-bundler v2 size limits.
# In multi-arch ASAN builds (e.g. the multi_arch_ci_asan workflow), the
# super-project passes the full family list (gfx906;gfx908;gfx90a;
# gfx942:xnack+;gfx950:xnack+) which would trip that check and break the
# whole build at the RCCL configure step. See:
#   https://github.com/ROCm/TheRock/issues/4770
#
# Detect that case and skip BUILD_ADDRESS_SANITIZER for RCCL only. Host-side
# ASAN flags (-fsanitize=address) are still injected globally by
# therock_sanitizers.cmake, so the host parts of librccl remain instrumented.
if(THEROCK_SANITIZER STREQUAL "ASAN")
  list(LENGTH GPU_TARGETS _therock_rccl_num_gpu_targets)
  if(_therock_rccl_num_gpu_targets GREATER 1)
    message(WARNING
      "Skipping BUILD_ADDRESS_SANITIZER for RCCL because GPU_TARGETS contains "
      "${_therock_rccl_num_gpu_targets} entries (${GPU_TARGETS}). "
      "RCCL's ASAN device-code path only supports a single GPU_TARGET due to "
      "clang-offload-bundler size limits. Host-side ASAN instrumentation is "
      "still applied via global compiler flags. To enable full device-side "
      "ASAN for RCCL, configure the build with a single GPU target.")
  else()
    set(BUILD_ADDRESS_SANITIZER ON)
    message(STATUS "Enabling BUILD_ADDRESS_SANITIZER for RCCL "
      "(THEROCK_SANITIZER=${THEROCK_SANITIZER}, GPU_TARGETS=${GPU_TARGETS})")
  endif()
endif()
