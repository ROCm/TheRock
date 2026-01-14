# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(hipdnn_install_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

# Define tests: package_name:header:test_suffix
set(HIPDNN_PACKAGES
    "hipdnn_backend:hipdnn_backend.h:backend"
    "hipdnn_data_sdk:hipdnn_data_sdk/data_objects/data_types_generated.h:data_sdk"
    "hipdnn_frontend:hipdnn_frontend.hpp:frontend"
    "hipdnn_plugin_sdk:hipdnn_plugin_sdk/PluginApi.h:plugin_sdk"
    "hipdnn_test_sdk:hipdnn_test_sdk/utilities/LoggingUtils.hpp:test_sdk"
)

foreach(entry IN LISTS HIPDNN_PACKAGES)
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 pkg_name)
    list(GET parts 1 header)
    list(GET parts 2 test_suffix)

    find_package(${pkg_name} CONFIG REQUIRED)
    message(STATUS "${pkg_name} found")

    set(test_name "test_${test_suffix}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/test_template.cpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp"
        @ONLY
    )
    add_executable(${test_name} "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
    target_link_libraries(${test_name} PRIVATE ${pkg_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
