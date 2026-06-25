# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile is only built on gfx90a, gfx942, gfx950, gfx11 and gfx12
if(GPU_TARGETS MATCHES "gfx942|gfx950|gfx90a|gfx11|gfx12")

    function(add_moe_sorting_test EXECUTABLE USE_2D_BUF)
        add_gtest_executable(${EXECUTABLE} test_moe_sorting.cpp moe_sorting_api.cpp)
        target_include_directories(${EXECUTABLE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

        set(EXAMPLE_MOE_SORTING_COMPILE_OPTIONS)
        # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
        list(APPEND EXAMPLE_MOE_SORTING_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal -DMOE_SORTING_FMOE_2D_BUF=${USE_2D_BUF})
        target_compile_options(${EXECUTABLE} PRIVATE ${EXAMPLE_MOE_SORTING_COMPILE_OPTIONS})

    endfunction(add_moe_sorting_test EXECUTABLE USE_2D_BUF)

    add_moe_sorting_test(test_ck_tile_moe_sorting_2d_buf 1)
    add_moe_sorting_test(test_ck_tile_moe_sorting 0)

else()
    message(DEBUG "Skipping ck_tile_moe_sorting tests for current target")
endif()
