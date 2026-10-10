# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/therock_sanitizers.cmake")
therock_sanitizer_configure(stanza selected "${CMAKE_CXX_COMPILER}" TRUE fixture)
file(WRITE "${CMAKE_BINARY_DIR}/sanitizer.cmake" "${stanza}")
include("${CMAKE_BINARY_DIR}/sanitizer.cmake")
project(host_sanitizer LANGUAGES C CXX HIP)
find_package(Threads REQUIRED)

add_library(sanitizer_probe SHARED probe.c probe.cpp probe.hip)
add_executable(sanitizer_control control.cpp)
target_compile_features(sanitizer_control PRIVATE cxx_std_17)
target_link_libraries(sanitizer_control PRIVATE sanitizer_probe Threads::Threads)
