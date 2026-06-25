# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_example_topk_softmax topk_softmax.cpp topk_softmax_api.cpp)
target_include_directories(tile_example_topk_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

set(EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS)
# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(tile_example_topk_softmax PRIVATE ${EXAMPLE_TOPK_SOFTMAX_COMPILE_OPTIONS})
