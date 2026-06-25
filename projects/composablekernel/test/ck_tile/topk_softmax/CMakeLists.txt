# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(add_tile_topk_softmax_test SUFFIX)
    set(TEST_NAME "test_ck_tile_topk_softmax_${SUFFIX}")
    add_test_executable(${TEST_NAME} test_topk_softmax_${SUFFIX}.cpp test_topk_softmax_api.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

    set(TEST_TOPK_SOFTMAX_COMPILE_OPTIONS)
    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND TEST_TOPK_SOFTMAX_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    # list(APPEND TEST_TOPK_SOFTMAX_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
    target_compile_options(${TEST_NAME} PRIVATE ${TEST_TOPK_SOFTMAX_COMPILE_OPTIONS})
endfunction()

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_tile_topk_softmax_test(fp16)
    add_tile_topk_softmax_test(bf16)
else()
    message(DEBUG "Skipping tile topk_softmax tests for current target")
endif()
