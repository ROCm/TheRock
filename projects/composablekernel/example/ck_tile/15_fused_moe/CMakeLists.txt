# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95")
  set(TILE_EXAMPLE_FUSED_MOE "tile_example_fused_moe")
  message(DEBUG "adding ${TILE_EXAMPLE_FUSED_MOE}")
  file(GLOB INSTANCE_SRCS instances/*.cpp)
  add_executable(${TILE_EXAMPLE_FUSED_MOE} main.cpp)
  target_include_directories(${TILE_EXAMPLE_FUSED_MOE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_sources(${TILE_EXAMPLE_FUSED_MOE} PRIVATE ${INSTANCE_SRCS})

  set(TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS)

  # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
  list(APPEND TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
  list(APPEND TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_BUFFER_LOAD_AGPR=1) # TODO: enable load to a
  list(APPEND TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=4) # rta
  # list(APPEND TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS  -mllvm -greedy-reverse-local-assignment=1)
  # list(APPEND TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

  target_compile_options(${TILE_EXAMPLE_FUSED_MOE} PRIVATE ${TILE_EXAMPLE_FUSED_MOE_COMPILE_OPTIONS})
endif()
