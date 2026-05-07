# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_WMMA_KERNELS
set(GROUPED_CONV3D_BWD_WEIGHT_SCALE
    xdl/device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
    xdl/device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
    xdl/device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.cpp
    xdl/device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp)

if((DTYPES MATCHES "fp8" AND DTYPES MATCHES "bf8" AND DTYPES MATCHES "fp16") OR NOT DEFINED DTYPES)
    list(APPEND GROUPED_CONV3D_BWD_WEIGHT_SCALE
      xdl/device_grouped_conv3d_bwd_weight_xdl_scale_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_fp8_instance.cpp)
endif()

list(APPEND GROUPED_CONV3D_BWD_WEIGHT_SCALE
    wmma/device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
    wmma/device_grouped_conv3d_bwd_weight_wmma_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
    )

add_instance_library(device_grouped_conv3d_bwd_weight_scale_instance ${GROUPED_CONV3D_BWD_WEIGHT_SCALE})
