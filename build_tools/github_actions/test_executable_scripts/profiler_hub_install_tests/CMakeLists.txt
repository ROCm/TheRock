# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(profiler_hub_install_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

function(add_profiler_hub_install_test)
    cmake_parse_arguments(ARG "" "PACKAGE;TARGET;HEADER" "" ${ARGN})

    find_package(${ARG_PACKAGE} CONFIG REQUIRED)
    message(STATUS "${ARG_PACKAGE} found")

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
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_profiler_hub_install_test(
    PACKAGE profiler-hub
    TARGET profiler-hub::profiler-hub
    HEADER profiler-hub/storage.hpp
)
