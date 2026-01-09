# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test_plugin_sdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
message(STATUS "hipdnn_plugin_sdk found")

add_executable(test_plugin_sdk test_plugin_sdk.cpp)
target_link_libraries(test_plugin_sdk PRIVATE hipdnn_plugin_sdk)

enable_testing()
add_test(NAME test_plugin_sdk COMMAND test_plugin_sdk)
