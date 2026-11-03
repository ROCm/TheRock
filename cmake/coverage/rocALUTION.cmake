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

# hip_add_library() also links rocalution_hip as C++, and a C++ link with
# -fprofile-instr-generate gets only libclang_rt.profile.a. The instrumented
# kernels' registrations then take the GPU half of the profile runtime from
# it, without the sanitizer helpers that only libclang_rt.profile_rocm.a
# defines, and the library does not load. --hip-link has the driver link it
# as it links any HIP binary, with libclang_rt.profile_rocm.a. The target
# exists once the top-level CMakeLists.txt has run.
get_property(_therock_coverage_hip_link_deferred GLOBAL
  PROPERTY THEROCK_COVERAGE_ROCALUTION_HIP_LINK_DEFERRED)
if(NOT _therock_coverage_hip_link_deferred)
  set_property(GLOBAL PROPERTY THEROCK_COVERAGE_ROCALUTION_HIP_LINK_DEFERRED TRUE)
  function(_therock_coverage_rocalution_hip_link)
    if(TARGET rocalution_hip)
      target_link_options(rocalution_hip PRIVATE --hip-link)
    endif()
  endfunction()
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL _therock_coverage_rocalution_hip_link)
endif()
