# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Built here, shipped in the test component. Declared by profiler/CMakeLists.txt.

cmake_minimum_required(VERSION 3.25)
project(profiler-hub-install-test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT PROFILER_HUB_ARTIFACT_PREFIX)
  message(
    FATAL_ERROR
    "PROFILER_HUB_ARTIFACT_PREFIX must name an assembled profiler-hub artifact tree"
  )
endif()

function(add_profiler_hub_install_test)
  cmake_parse_arguments(ARG "" "PACKAGE;TARGET;HEADER" "" ${ARGN})

  find_package(${ARG_PACKAGE} CONFIG REQUIRED
    PATHS "${PROFILER_HUB_ARTIFACT_PREFIX}"
    NO_DEFAULT_PATH
  )
  message(STATUS "${ARG_PACKAGE} found in ${PROFILER_HUB_ARTIFACT_PREFIX}")

  set(test_name "test_${ARG_PACKAGE}")
  set(header "${ARG_HEADER}")
  set(pkg_name "${ARG_PACKAGE}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/test_template.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp"
    @ONLY
  )
  add_executable(${test_name} "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  target_link_libraries(${test_name} PRIVATE ${ARG_TARGET})
  install(TARGETS ${test_name} RUNTIME DESTINATION bin)
endfunction()

add_profiler_hub_install_test(
  PACKAGE profiler-hub
  TARGET profiler-hub::profiler-hub
  HEADER profiler-hub/storage.hpp
)
