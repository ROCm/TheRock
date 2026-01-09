# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test_frontend LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_frontend CONFIG REQUIRED)
message(STATUS "hipdnn_frontend found")

add_executable(test_frontend test_frontend.cpp)
target_link_libraries(test_frontend PRIVATE hipdnn_frontend)

enable_testing()
add_test(NAME test_frontend COMMAND test_frontend)
