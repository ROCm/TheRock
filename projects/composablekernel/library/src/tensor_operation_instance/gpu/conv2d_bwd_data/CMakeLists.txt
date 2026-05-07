# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_AND_DL_KERNELS
set(CONV2D_BWD_DATA_INSTANCES)
list(APPEND CONV2D_BWD_DATA_INSTANCES device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instance.cpp
                                      device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f32_instance.cpp
                                      device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp
                                      device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instance.cpp
                                      device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instance.cpp
                                      device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instance.cpp
                                      device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_int8_instance.cpp)

add_instance_library(device_conv2d_bwd_data_instance ${CONV2D_BWD_DATA_INSTANCES})
