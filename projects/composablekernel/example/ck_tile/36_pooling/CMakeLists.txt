# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_POOL_3D "tile_example_pool3d")
message(DEBUG "adding example ${EXAMPLE_POOL_3D}")

add_executable(${EXAMPLE_POOL_3D} pool3d.cpp)
target_include_directories(${EXAMPLE_POOL_3D} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(${EXAMPLE_POOL_3D} PRIVATE ${EXAMPLE_POOL_COMPILE_OPTIONS})
