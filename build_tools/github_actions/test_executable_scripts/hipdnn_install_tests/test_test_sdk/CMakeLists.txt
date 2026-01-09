# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test_test_sdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_test_sdk CONFIG REQUIRED)
message(STATUS "hipdnn_test_sdk found")

add_executable(test_test_sdk test_test_sdk.cpp)
target_link_libraries(test_test_sdk PRIVATE hipdnn_test_sdk)

enable_testing()
add_test(NAME test_test_sdk COMMAND test_test_sdk)
