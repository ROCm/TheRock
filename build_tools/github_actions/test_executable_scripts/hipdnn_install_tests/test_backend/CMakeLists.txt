# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test_backend LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_backend CONFIG REQUIRED)
message(STATUS "hipdnn_backend found")

add_executable(test_backend test_backend.cpp)
target_link_libraries(test_backend PRIVATE hipdnn_backend)

enable_testing()
add_test(NAME test_backend COMMAND test_backend)
