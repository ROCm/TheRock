# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
  set(EXAMPLE_CONV_COMPILE_OPTIONS)
  list(APPEND EXAMPLE_CONV_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

  add_executable(tile_example_grouped_conv_fwd grouped_convolution_forward.cpp)
  target_compile_options(tile_example_grouped_conv_fwd PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})

  add_executable(tile_example_grouped_conv_fwd_large_tensor grouped_convolution_forward_large_tensor.cpp)
  target_compile_options(tile_example_grouped_conv_fwd_large_tensor PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})

  add_executable(tile_example_grouped_conv_fwd_bias_clamp grouped_convolution_forward_bias_clamp.cpp)
  target_compile_options(tile_example_grouped_conv_fwd_bias_clamp PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

  add_executable(tile_example_grouped_conv_bwd_weight grouped_convolution_backward_weight.cpp)
  target_compile_options(tile_example_grouped_conv_bwd_weight PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})

  add_executable(tile_example_grouped_conv_bwd_weight_two_stage grouped_convolution_backward_weight_two_stage.cpp)
  target_compile_options(tile_example_grouped_conv_bwd_weight_two_stage PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})

  add_executable(tile_example_grouped_conv_bwd_data grouped_convolution_backward_data.cpp)
  target_compile_options(tile_example_grouped_conv_bwd_data PRIVATE ${EXAMPLE_CONV_COMPILE_OPTIONS})
endif()
