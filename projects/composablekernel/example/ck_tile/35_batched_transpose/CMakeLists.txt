# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a")
  set(TARGET_NAME tile_example_batched_transpose)
  add_executable(${TARGET_NAME} batched_transpose_example.cpp batched_transpose_api.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

  # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
  list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
  # list(APPEND EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
  target_compile_options(tile_example_batched_transpose PRIVATE ${EXAMPLE_BATCHED_TRANSPOSE_COMPILE_OPTIONS})
endif()
