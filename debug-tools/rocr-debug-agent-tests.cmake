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

#
# rocm-debug-agent tests have the following dependencies:
#
# rocr-debug-agent
# HIP compiler/tools
#
#
# Available defines to be used when building rocr-debug-agent-test:
#
# GPU_TARGETS (optional)
#
# Build tests from sources for a list of GPU targets.
#
# GPU_TARGETS lists one or more GPU targets. If set, build the test files
# for each entry in GPU target. If not set, fallback to building for
# whatever GPU target the tools can find on the system.
#
#
# ENABLE_TESTS
#
# Enable the rocr-debug-agent-test testing target. Default is OFF.
#
# As part of building the rocr-debug-agent tests, we also add a testing
# target that can be invoked to execute the tests. Alternatively the tests
# can be invoked by hand.

cmake_minimum_required(VERSION 3.12)

project(rocm-debug-agent-test)

#include(GNUInstallDirs)

set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Start by trying to locate the HIP module that will tell us where all the
# HIP dependencies are.
find_package(HIP MODULE QUIET)

# Check if we managed to set a particular variable that means our HIP module
# discovery call above worked.
if(NOT HIP_CXX_COMPILER)
  # We have not found the HIP module. Try doing it differently.

  # Try setting CMAKE_MODULE_PATH to the default installation path.
  set(CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip" CACHE STRING "HIP CMAKE Module Path")
  find_package(HIP MODULE QUIET)

  # Mandate correct CMAKE_MODULE_PATH to be provided if the HIP module could not
  # be found in the default installation path.
  if( NOT HIP_FOUND )
    message(FATAL_ERROR
            "HIP Module not found in Module Path ${CMAKE_MODULE_PATH}, \
             Please pass the correct Module search Path \
             'cmake -DCMAKE_MODULE_PATH=/opt/rocm-<version>/lib/cmake/hip'")
    return()
  endif()
endif()

# Set the HIP language runtime link flags as FindHIP does not set them.
#set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
#set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
#set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_HIP_ARCHITECTURES OFF)

set(MAIN_SOURCES
        debug_agent_test.cpp
        vector_add_assert_trap.cpp
        sigquit.cpp
        print_all_waves.cpp
        snapshot_objfile_on_load.cpp
        vector_add_normal.cpp
        vector_add_memory_fault.cpp
        save_code_objects.cpp)
set(NO_DEBUG_SOURCES
        vector_add_assert_trap_no_debug_info.cpp)
set(SOURCES ${MAIN_SOURCES} ${NO_DEBUG_SOURCES})

set_source_files_properties(${SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

# If we have a list of GPU targets, cycle through them and build tests for
# each specific target.
if (GPU_TARGETS)
  foreach(ARCH_TARGET IN LISTS GPU_TARGETS)
    message(STATUS "Setting up tests for GPU target: ${ARCH_TARGET}")

    # Define a variable for the target-specific binary directory
    set(ARCH_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${ARCH_TARGET})

    # Use the correct file command to create the directory
    file(MAKE_DIRECTORY ${ARCH_BIN_DIR})

    add_custom_command(
      OUTPUT "${ARCH_BIN_DIR}/snapshot_objfile_on_load.hipfb"
      COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/snapshot_objfile_on_load.cpp --genco --offload-arch=${ARCH_TARGET} -o ${ARCH_BIN_DIR}/snapshot_objfile_on_load.hipfb
      DEPENDS "snapshot_objfile_on_load.cpp"
      COMMENT "Building snapshot_objfile_on_load.hipfb for ${ARCH_TARGET}"
      VERBATIM)

    add_custom_command(
      OUTPUT "${ARCH_BIN_DIR}/save_code_objects.hipfb"
      COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/save_code_objects.cpp --genco --offload-arch=${ARCH_TARGET} -o ${ARCH_BIN_DIR}/save_code_objects.hipfb
      DEPENDS "save_code_objects.cpp"
      COMMENT "Building save_code_objects.hipfb for ${ARCH_TARGET}"
      VERBATIM)

    # Define targets to manage dependencies
    add_custom_target(snapshot_objfile_on_load-tgt-${ARCH_TARGET} ALL DEPENDS ${ARCH_BIN_DIR}/snapshot_objfile_on_load.hipfb)
    add_custom_target(save_code_objects-tgt-${ARCH_TARGET} ALL DEPENDS ${ARCH_BIN_DIR}/save_code_objects.hipfb)

    hip_add_library(rocm-debug-agent-no-dbgapifo-${ARCH_TARGET} ${NO_DEBUG_SOURCES})

    hip_add_executable(rocm-debug-agent-test-${ARCH_TARGET} ${MAIN_SOURCES} ${NODEBUG_OBJ} CLANG_OPTIONS -ggdb)
    target_link_libraries(rocm-debug-agent-test-${ARCH_TARGET} rocm-debug-agent-no-dbgapifo-${ARCH_TARGET})

    add_dependencies(rocm-debug-agent-test-${ARCH_TARGET} snapshot_objfile_on_load-tgt-${ARCH_TARGET} save_code_objects-tgt-${ARCH_TARGET})
  endforeach()
else()
  message(STATUS "Setting up tests for system GPU target")

  add_custom_command(
    OUTPUT "snapshot_objfile_on_load.hipfb"
    COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/snapshot_objfile_on_load.cpp --genco -o snapshot_objfile_on_load.hipfb
    DEPENDS "snapshot_objfile_on_load.cpp"
    COMMENT "Building snapshot_objfile_on_load.hipfb for system GPU"
    VERBATIM)

  add_custom_command(
    OUTPUT "save_code_objects.hipfb"
    COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/save_code_objects.cpp --genco -o save_code_objects.hipfb
    DEPENDS "save_code_objects.cpp"
    COMMENT "Building save_code_objects.hipfb for system GPU"
    VERBATIM)

  # Define targets to manage dependencies
  add_custom_target(snapshot_objfile_on_load-tgt ALL DEPENDS snapshot_objfile_on_load.hipfb)
  add_custom_target(save_code_objects-tgt ALL DEPENDS save_code_objects.hipfb)

  hip_add_library(rocm-debug-agent-no-dbgapifo ${NO_DEBUG_SOURCES})

  hip_add_executable(rocm-debug-agent-test ${MAIN_SOURCES} ${NODEBUG_OBJ} CLANG_OPTIONS -ggdb)
  target_link_libraries(rocm-debug-agent-test rocm-debug-agent-no-dbgapifo)

  add_dependencies(rocm-debug-agent-test snapshot_objfile_on_load-tgt save_code_objects-tgt)
endif()

file(GLOB HEADERS "*.h")

install(FILES CMakeLists.txt run-test.py ${SOURCES} ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/src/rocm-debug-agent-test
  COMPONENT tests)

if(ENABLE_TESTS)
  enable_testing()
  add_test(NAME rocm-debug-agent-test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run-test.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
