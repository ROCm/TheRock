################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

# Available defines to be used when building rocr-debug-agent:
#
# ENABLE_TESTS
#
# Enable building the rocr-debug-agent tests. Default is ON.
#
# The debug agent has a small set of validation tests that can be built
# with a suitable HIP compiler. This is mostly for development, so it isn't
# required to be built if users are only looking for the tool.
#
#
# rocm-debug-agent has the following dependencies:
#
# amd_dbgapi
# LibElf
# LibDW
# hsa-runtime64
# GIT (optional)
# Threads

cmake_minimum_required(VERSION 3.10)

project(rocm-debug-agent VERSION 2.1.0)

include(GNUInstallDirs)

file(GLOB SOURCES "src/*.cpp")
add_library(rocm-debug-agent SHARED ${SOURCES})

set_target_properties(rocm-debug-agent PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  OUTPUT_NAME "rocm-debug-agent"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  NO_SYSTEM_FROM_IMPORTED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# Start by looking for libdw (lowercase).
find_package(libdw CONFIG)

# If libdw is found, we're likely building within TheRock build platform.
#
# Note that we use the same CMAKE variables as we would use in a standalone
# build, to avoid duplication of code.
if(libdw_FOUND)
  # Set the include file path for libdw.
  get_property(LIBDW_INCLUDES
               TARGET libdw::libdw
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  # Set the libdw library reference accordingly for the link setup.
  set(LIBDW_LIBRARIES libdw::libdw)

  # Find the ROCm Runtime.
  find_package(hsa-runtime64 REQUIRED CONFIG PATHS "/opt/rocm")
  # Set the include file path for the ROCm Runtime.
  get_property(ROCR_INCLUDES
               TARGET hsa-runtime64::hsa-runtime64
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  # Set the ROCm Runtime library reference accordingly for the link setup.
  set(ROCR_LIBRARIES hsa-runtime64::hsa-runtime64)

  # Find LibElf.
  find_package(LibElf REQUIRED CONFIG)
  # Set the include file path for LibElf.
  get_property(LIBELF_INCLUDES
               TARGET elf::elf
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  # Set the LibElf library reference accordingly for the link setup.
  set(LIBELF_LIBRARIES elf::elf)
else()
  # Otherwise we're doing a regular standalone build, so use our cmake
  # modules to locate the ROCR and LibElf.
  find_package(ROCR REQUIRED)
  find_package(LibDw REQUIRED)
  find_package(LibElf REQUIRED)
endif()

message(STATUS "libdw include: ${LIBDW_INCLUDES}")
message(STATUS "libhsa include: ${ROCR_INCLUDES}")
message(STATUS "libelf include: ${LIBELF_INCLUDES}")

find_package(amd-dbgapi REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd-dbgapi
    lib/cmake/amd-dbgapi
)

find_package(Threads REQUIRED)

target_include_directories(rocm-debug-agent
  SYSTEM PRIVATE ${ROCR_INCLUDES} ${LIBELF_INCLUDES} ${LIBDW_INCLUDES})
target_compile_options(rocm-debug-agent PRIVATE -Werror -Wall)

if(DEFINED ENV{ROCM_BUILD_ID})
  # ROCM_BUILD_ID is set by the ROCm-CI build environment.
  set(build_info $ENV{ROCM_BUILD_ID})
else()
  string(TIMESTAMP NOW "%Y%m%dT%H%M%S")
  set(build_info developer-build-${NOW})

  if(DEFINED ENV{USER})
    set(build_info ${build_info}-$ENV{USER})
  endif()

  find_package(Git)
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE build_revision
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    message(STATUS "GIT not found")
  endif()

  if(DEFINED build_revision)
    set(build_info ${build_info}-git-${build_revision})
  endif()
endif()

# Check for memfd_create headers and libs
include(CheckCSourceCompiles)
check_c_source_compiles("
  #define _GNU_SOURCE
  #include <sys/mman.h>
  int main() { return memfd_create (\"cmake_test\", 0); }
  " HAVE_MEMFD_CREATE)
if (HAVE_MEMFD_CREATE)
  add_definitions(-DHAVE_MEMFD_CREATE)
endif()

target_link_libraries(rocm-debug-agent
  PRIVATE amd-dbgapi ${ROCR_LIBRARIES} ${LIBELF_LIBRARIES} ${LIBDW_LIBRARIES}
  Threads::Threads ${CMAKE_DL_LIBS}
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap -Wl,--no-undefined)

target_compile_options(rocm-debug-agent
  PRIVATE -fno-rtti -Wall -Wno-attributes -fvisibility=hidden)

target_compile_definitions(rocm-debug-agent
  PRIVATE AMD_INTERNAL_BUILD _GNU_SOURCE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)

install(TARGETS rocm-debug-agent
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT runtime)

install(FILES LICENSE.txt README.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT runtime)
# ASAN package requires libraries and license file
install(TARGETS rocm-debug-agent
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT asan)

install(FILES LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan
  COMPONENT asan)

# Add packaging directives for rocm-debug-agent
set(CPACK_PACKAGE_NAME rocm-debug-agent)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_PACKAGE_CONTACT "ROCm Debugger Support <rocm-gdb.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute Debug Agent (ROCdebug-agent)")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "rocm-debug-agent")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-dbgapi, hsa-rocr, rocm-core")
set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "rocm-debug-agent, rocm-core")

# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "rocm-debug-agent-asan")
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-dbgapi-asan, hsa-rocr-asan, rocm-core-asan")
# Debian package specific variables
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE ${build_info})
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "rocm-debug-agent")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-dbgapi, hsa-rocr, rocm-core")
set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "rocm-debug-agent, rocm-core")

# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "rocm-debug-agent-asan")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-dbgapi-asan, hsa-rocr-asan, rocm-core-asan")
# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_TESTS_PACKAGE_REQUIRES ${CPACK_RPM_TESTS_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS ${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
endif()

# By default build the Debug Agent tests.
option(ENABLE_TESTS "Enable building Debug Agent tests" ON)

set(TEST_COMPONENT "tests")

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
else()
  set(TEST_COMPONENT)
endif()

if(ENABLE_ASAN_PACKAGING)
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL runtime ${TEST_COMPONENT})
endif()

if (NOT CPack_CMake_INCLUDED)
  include(CPack)
endif()

cpack_add_component(runtime
  DISPLAY_NAME "Runtime"
  DESCRIPTION "Dynamic libraries for the ROCdebug-agent")

if(ENABLE_TESTS)
  cpack_add_component(tests
    DISPLAY_NAME "Tests"
    DESCRIPTION "Tests for the ROCdebug-agent"
    DEPENDS runtime)
endif()

cpack_add_component(asan
  DISPLAY_NAME "Asan"
  DESCRIPTION "AddressSanitizer libraries for the ROCdebug-agent")
