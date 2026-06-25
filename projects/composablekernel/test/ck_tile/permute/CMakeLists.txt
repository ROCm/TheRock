# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")

    function(add_permute_test TARGET_NAME MAIN_SRC)
        add_gtest_executable(${TARGET_NAME} ${MAIN_SRC})

        if(NOT DEFINED PERMUTE_USE_ALTERNATIVE_IMPL)
        set(PERMUTE_USE_ALTERNATIVE_IMPL true)
        endif()

        if(PERMUTE_USE_ALTERNATIVE_IMPL)
        target_compile_options(${TARGET_NAME} PRIVATE -DPERMUTE_USE_ALTERNATIVE_IMPL)
        endif()

    endfunction(add_permute_test TARGET_NAME MAIN_SRC)

    add_permute_test(test_ck_tile_permute test_permute.cpp)

else()
    message(DEBUG "Skipping ck_tile_permute tests for current target")
endif()
