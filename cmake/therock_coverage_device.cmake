# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Included by the CMAKE_PROJECT_INCLUDE file that therock_subproject.cmake
# generates for a subproject built with <PROJECT>_ENABLE_DEVICE_COVERAGE, which
# that file passes in as THEROCK_COVERAGE_DEVICE_OPTION. It runs after every
# project() call: the runtime lookups happen once, as soon as a compiler is
# known, and the link rule rewrite below is repeated for each new language.
#
# Fails the configure when the compiler has no amdgcn profile runtime. The
# linker wrapper forwards the profile flags to the device link only when that
# runtime exists, while the device compile is instrumented regardless, so
# without this check the problem surfaces later and less clearly: as a device
# link error, or as a report with no device coverage in it.
#
# Device counters are read back at exit only by the InstrProfilingFile.o in
# libclang_rt.profile_rocm.a; the copy in libclang_rt.profile.a has that call
# compiled out. The driver links profile_rocm ahead of profile on a HIP link
# with -fprofile-instr-generate, but a project that links the runtime itself
# can put them the other way round (rocSPARSE does, to keep the profile flags
# off its link), and then its kernels are counted and never written. So
# profile_rocm goes in front of <LINK_LIBRARIES> in every link rule, where it
# resolves the runtime before anything the project names; a link with no
# instrumented objects pulls nothing from it.

if(NOT DEFINED THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME
   OR NOT DEFINED THEROCK_COVERAGE_HOST_PROFILE_RUNTIME)
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

  # Per-target runtime layout first, then the arch-suffixed one used by
  # compilers built without LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
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

  foreach(_therock_coverage_name IN ITEMS
      "libclang_rt.profile_rocm.a"
      "libclang_rt.profile_rocm-${CMAKE_SYSTEM_PROCESSOR}.a")
    execute_process(
      COMMAND "${_therock_coverage_compiler}" "-print-file-name=${_therock_coverage_name}"
      OUTPUT_VARIABLE _therock_coverage_candidate
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(IS_ABSOLUTE "${_therock_coverage_candidate}" AND EXISTS "${_therock_coverage_candidate}")
      set(THEROCK_COVERAGE_HOST_PROFILE_RUNTIME "${_therock_coverage_candidate}"
        CACHE INTERNAL "Host profile runtime that reads device counters back at exit")
      break()
    endif()
  endforeach()
  if(NOT DEFINED THEROCK_COVERAGE_HOST_PROFILE_RUNTIME)
    message(FATAL_ERROR
      "Device coverage is enabled for ${PROJECT_NAME}, but the compiler "
      "'${_therock_coverage_compiler}' has no libclang_rt.profile_rocm.a, the "
      "host runtime that reads device counters back. Build compiler-rt with "
      "COMPILER_RT_BUILD_PROFILE_ROCM, or configure TheRock with "
      "-D${THEROCK_COVERAGE_DEVICE_OPTION}=OFF for host-only coverage.")
  endif()

  message(STATUS
    "Device coverage: amdgcn profile runtime ${THEROCK_COVERAGE_DEVICE_PROFILE_RUNTIME}, "
    "host collector ${THEROCK_COVERAGE_HOST_PROFILE_RUNTIME}")
endif()

foreach(_therock_coverage_lang IN ITEMS C CXX HIP)
  foreach(_therock_coverage_rule IN ITEMS
      CREATE_SHARED_LIBRARY CREATE_SHARED_MODULE LINK_EXECUTABLE)
    set(_therock_coverage_var
      "CMAKE_${_therock_coverage_lang}_${_therock_coverage_rule}")
    if(${_therock_coverage_var}
       AND NOT "${${_therock_coverage_var}}" MATCHES "clang_rt\\.profile_rocm")
      string(REPLACE "<LINK_LIBRARIES>"
        "\"${THEROCK_COVERAGE_HOST_PROFILE_RUNTIME}\" <LINK_LIBRARIES>"
        ${_therock_coverage_var} "${${_therock_coverage_var}}")
    endif()
  endforeach()
endforeach()
