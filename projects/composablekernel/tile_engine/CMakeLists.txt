# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

add_subdirectory(ops/gemm)
add_subdirectory(ops/gemm_streamk)
add_subdirectory(ops/reduce)

