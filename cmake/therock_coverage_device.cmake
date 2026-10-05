# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Included by the CMAKE_PROJECT_INCLUDE file that therock_subproject.cmake
# generates for a subproject built with <PROJECT>_ENABLE_DEVICE_COVERAGE, which
# that file passes in as THEROCK_COVERAGE_DEVICE_OPTION. It runs after every
# project() call, so this does its work once, as soon as a compiler is known.
#
# Fails the configure when the compiler has no amdgcn profile runtime. The
# linker wrapper forwards the profile flags to the device link only when that
# runtime exists, while the device compile is instrumented regardless, so
# without this check the problem surfaces later and less clearly: as a device
# link error, or as a report with no device coverage in it.

if(DEFINED THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME)
  return()
endif()

set(_therock_coverage_compiler "${CMAKE_CXX_COMPILER}")
if(NOT _therock_coverage_compiler)
  set(_therock_coverage_compiler "${CMAKE_HIP_COMPILER}")
endif()
if(NOT _therock_coverage_compiler)
  return()
endif()

execute_process(
  COMMAND "${_therock_coverage_compiler}" --target=amdgcn-amd-amdhsa -print-resource-dir
  OUTPUT_VARIABLE _therock_coverage_resource_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

# Per-target runtime layout first, then the arch-suffixed one used by compilers
# built without LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
foreach(_therock_coverage_relpath IN ITEMS
    "lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
    "lib/linux/libclang_rt.profile-amdgcn.a")
  set(_therock_coverage_candidate
    "${_therock_coverage_resource_dir}/${_therock_coverage_relpath}")
  if(_therock_coverage_resource_dir AND EXISTS "${_therock_coverage_candidate}")
    set(THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME "${_therock_coverage_candidate}"
      CACHE INTERNAL "amdgcn profile runtime the device coverage build links")
    break()
  endif()
endforeach()

if(NOT DEFINED THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME)
  message(FATAL_ERROR
    "Device coverage is enabled for ${PROJECT_NAME}, but the compiler "
    "'${_therock_coverage_compiler}' has no amdgcn profile runtime under "
    "'${_therock_coverage_resource_dir}'. Build the compiler with "
    "COMPILER_RT_BUILD_PROFILE for the amdgcn-amd-amdhsa runtimes target, or "
    "configure TheRock with -D${THEROCK_COVERAGE_DEVICE_OPTION}=OFF for "
    "host-only coverage.")
endif()
message(STATUS
  "Device coverage: amdgcn profile runtime ${THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME}")
