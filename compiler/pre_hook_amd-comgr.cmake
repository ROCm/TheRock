# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Enable ASAN for Comgr when THEROCK_SANITIZER is set to ASAN or HOST_ASAN
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  set(ADDRESS_SANITIZER ON)
  message(STATUS "Enabling ASAN for Comgr (THEROCK_SANITIZER=${THEROCK_SANITIZER})")
endif()

if(THEROCK_BUILD_COMGR_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "Enable comgr tests" FORCE)
else()
  set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")

# Windows Comgr DLL naming.
#
# CLR's Comgr::LoadLib loads the Comgr DLL by an exact filename. The current
# CLR code computes a HIP-versioned name at runtime (amd_comgr0702.dll), so
# Comgr must produce a DLL with that name.
#
# After llvm-project#1278 (merged), Comgr reads COMGR_DLL_NAME as a cache
# variable. After rocm-systems#4211 (pending), CLR will also read
# COMGR_DLL_NAME at build time. Once both submodules include those changes,
# COMGR_DLL_NAME can be removed here (both default to amd_comgr.dll).
if(WIN32 AND DEFINED THEROCK_HIP_MAJOR_VERSION AND DEFINED THEROCK_HIP_MINOR_VERSION)
  set(_comgr_major "${THEROCK_HIP_MAJOR_VERSION}")
  set(_comgr_minor "${THEROCK_HIP_MINOR_VERSION}")
  if(_comgr_major LESS_EQUAL 9)
    set(_comgr_major "0${_comgr_major}")
  endif()
  if(_comgr_minor LESS_EQUAL 9)
    set(_comgr_minor "0${_comgr_minor}")
  endif()

  set(_comgr_dll_name "amd_comgr${_comgr_major}${_comgr_minor}.dll")

  # For new Comgr (post llvm-project#1278): sets the DLL name via cache var.
  set(COMGR_DLL_NAME "${_comgr_dll_name}" CACHE STRING
    "Windows Comgr DLL output name" FORCE)

  # For old Comgr (pre llvm-project#1278): override OUTPUT_NAME after the
  # subproject's CMakeLists.txt runs. Old Comgr ignores the cache var and
  # unconditionally sets OUTPUT_NAME to amd_comgr_${VERSION_MAJOR}.
  # TODO: Remove this deferred call once llvm-project submodule includes #1278.
  function(_therock_comgr_output_name_fallback)
    # Read _comgr_dll_name (not COMGR_DLL_NAME) because old Comgr's
    # non-cache set(COMGR_DLL_NAME ...) shadows our cache variable.
    string(REGEX REPLACE "\\.dll$" "" _name "${_comgr_dll_name}")
    message(STATUS "Override comgr OUTPUT_NAME (windows): ${_name}")
    set_target_properties(amd_comgr PROPERTIES OUTPUT_NAME "${_name}")
  endfunction()
  cmake_language(DEFER CALL _therock_comgr_output_name_fallback)
endif()
