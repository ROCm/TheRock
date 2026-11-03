# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Included by the CMAKE_PROJECT_INCLUDE file that therock_subproject.cmake
# generates for a subproject built with <PROJECT>_ENABLE_DEVICE_COVERAGE, which
# that file passes in as THEROCK_COVERAGE_DEVICE_OPTION. It runs after every
# project() call; the runtime lookups happen once, as soon as a compiler is
# known.
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
# with -fprofile-instr-generate, but a project that names the runtime itself
# can put them the other way round (rocSPARSE does, to keep the profile flags
# off its link), and then its kernels are counted and never written. Once the
# project has defined its targets, profile_rocm is inserted in front of the
# generic archive in any shared library that names it.
#
# Only there: profile_rocm also defines hipLaunchKernel and the other launch
# calls as interceptors, so in the link rules it would resolve those calls
# ahead of the HIP runtime in every binary, instrumented or not. Executables
# that name the generic archive, as rocSPARSE's unit tests do, keep it.

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

# Puts the collector ahead of the generic profile archive wherever a shared
# library in dir, or below it, names that archive before (or without) the
# collector.
function(_therock_coverage_link_collector_first dir)
  get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _targets)
    get_target_property(_type "${_target}" TYPE)
    if(NOT _type MATCHES "^(SHARED|MODULE)_LIBRARY$")
      continue()
    endif()
    get_target_property(_libs "${_target}" LINK_LIBRARIES)
    if(NOT _libs)
      continue()
    endif()
    set(_generic -1)
    set(_collector -1)
    set(_index 0)
    foreach(_lib IN LISTS _libs)
      if(_generic EQUAL -1 AND _lib MATCHES "libclang_rt\\.profile(-[^/]+)?\\.a$")
        set(_generic ${_index})
      elseif(_collector EQUAL -1 AND _lib MATCHES "libclang_rt\\.profile_rocm")
        set(_collector ${_index})
      endif()
      math(EXPR _index "${_index} + 1")
    endforeach()
    if(_generic EQUAL -1 OR (_collector GREATER -1 AND _collector LESS _generic))
      continue()
    endif()
    list(INSERT _libs ${_generic} "${THEROCK_COVERAGE_HOST_PROFILE_RUNTIME}")
    set_property(TARGET "${_target}" PROPERTY LINK_LIBRARIES "${_libs}")
    message(STATUS
      "Device coverage: ${_target} links libclang_rt.profile.a itself; "
      "linking libclang_rt.profile_rocm.a ahead of it")
  endforeach()
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_subdir IN LISTS _subdirs)
    _therock_coverage_link_collector_first("${_subdir}")
  endforeach()
endfunction()

get_property(_therock_coverage_deferred GLOBAL PROPERTY THEROCK_COVERAGE_COLLECTOR_DEFERRED)
if(NOT _therock_coverage_deferred)
  set_property(GLOBAL PROPERTY THEROCK_COVERAGE_COLLECTOR_DEFERRED TRUE)
  # At the end of the top-level CMakeLists.txt, every target exists.
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL _therock_coverage_link_collector_first "${CMAKE_SOURCE_DIR}")
endif()
