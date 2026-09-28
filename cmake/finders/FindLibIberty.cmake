# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This finder is injected at the front of CMAKE_MODULE_PATH for all TheRock
# sub-projects via therock_subproject.cmake.
#
# When LibIberty is bundled (present in THEROCK_PROVIDED_PACKAGES), the
# therock_subproject_dep_provider intercepts find_package(LibIberty) before
# this module is reached and loads libiberty-config.cmake directly.  That
# config file creates the LibIberty::LibIberty SHARED IMPORTED GLOBAL target.
#
# When LibIberty is not bundled, the dep provider falls back here and we
# report not-found so that the sub-project (rocprofiler-systems / Dyninst)
# can use its own build mechanism (ROCPROFSYS_BUILD_LIBIBERTY=ON).
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("LibIberty" IN_LIST THEROCK_PROVIDED_PACKAGES)
  message(STATUS "Resolving bundled libiberty from super-project")
  find_package(LibIberty CONFIG REQUIRED)
else()
  set(LibIberty_FOUND FALSE)
endif()

cmake_policy(POP)
