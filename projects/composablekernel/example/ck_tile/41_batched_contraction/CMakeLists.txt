# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_example_batched_contraction batched_contraction.cpp)
set(EXAMPLE_CONTRACTION_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_CONTRACTION_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(tile_example_batched_contraction PRIVATE ${EXAMPLE_CONTRACTION_COMPILE_OPTIONS})
