# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test_data_sdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_data_sdk CONFIG REQUIRED)
message(STATUS "hipdnn_data_sdk found")

add_executable(test_data_sdk test_data_sdk.cpp)
target_link_libraries(test_data_sdk PRIVATE hipdnn_data_sdk)

enable_testing()
add_test(NAME test_data_sdk COMMAND test_data_sdk)
