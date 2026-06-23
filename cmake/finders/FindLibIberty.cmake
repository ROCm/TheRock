# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This finder resolves LibIberty for sub-projects when TheRock bundles libiberty
# as part of the binutils sysdep. It is injected at the front of CMAKE_MODULE_PATH
# for all subprojects, so it is found before rocprofiler-systems own
# FindLibIberty.cmake and redirects the lookup to the TheRock-managed config
# package installed under lib/rocm_sysdeps.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("LibIberty" IN_LIST THEROCK_PROVIDED_PACKAGES)
  message(STATUS "Resolving bundled libiberty from super-project")
  find_package(LibIberty CONFIG REQUIRED)
  # rocprofiler-systems references LibIberty::LibIberty; alias it to the
  # canonical lowercase target created by the config file.
  if(TARGET libiberty::libiberty AND NOT TARGET LibIberty::LibIberty)
    add_library(LibIberty::LibIberty ALIAS libiberty::libiberty)
  endif()
  cmake_policy(POP)
else()
  cmake_policy(POP)
  set(LibIberty_FOUND FALSE)
endif()
