# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Included by the CMAKE_PROJECT_INCLUDE file that therock_subproject.cmake
# generates for rocALUTION while its coverage is on. Unless USE_HIPCXX is on,
# rocALUTION builds its HIP backend with FindHIP's hip_add_library(), which
# compiles each source in a custom command running hipcc on HIP_HIPCC_FLAGS and
# then HIP_CLANG_FLAGS. The compile rules that carry the coverage flags never
# see those sources, so librocalution_hip.so would be built uninstrumented,
# kernels and host code alike. FindHIP creates both variables as empty cache
# entries, and under rocALUTION's policy settings (CMP0126 unset) creating one
# also drops a normal variable of the same name, so the flags go into the cache
# entry ahead of it.

if(NOT HIP_CLANG_FLAGS MATCHES "fcoverage-mapping")
  separate_arguments(_therock_coverage_flags UNIX_COMMAND
    "${THEROCK_COVERAGE_COMPILE_FLAGS}")
  set(HIP_CLANG_FLAGS ${HIP_CLANG_FLAGS} ${_therock_coverage_flags}
    CACHE STRING "Semicolon delimited flags for CLANG" FORCE)
endif()
