# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if (NOT GPU_TARGETS MATCHES "gfx11")
    add_custom_target(example_convnd_activ_xdl_convscale_relu)
    add_example_executable(example_convnd_fwd_xdl_convscale_relu_fp8 convnd_fwd_xdl_convscale_relu_fp8.cpp)
    add_example_dependencies(example_convnd_activ_xdl_convscale_relu example_convnd_fwd_xdl_convscale_relu_fp8)
endif()
