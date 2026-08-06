# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Minimal downstream consumer of the installed profiler-hub CMake package.
# Exercises find_package resolution, header availability, and linking against
# the installed profiler-hub::profiler-hub imported target. Configure with
# CMAKE_PREFIX_PATH pointed at the profiler-hub install prefix (the directory
# containing lib/cmake/profiler-hub), not the profiler-hub build tree.

cmake_minimum_required(VERSION 3.21)
project(profiler_hub_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(profiler-hub CONFIG REQUIRED)

add_executable(profiler_hub_consumer consumer.cpp)
target_link_libraries(profiler_hub_consumer PRIVATE profiler-hub::profiler-hub)
