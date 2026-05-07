# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(DEVICE_CONV2D_FWD_INSTANCES)
list(APPEND DEVICE_CONV2D_FWD_INSTANCES device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instance.cpp
                                        device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instance.cpp
                                        device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instance.cpp
                                        device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp
                                        device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instance.cpp)

add_instance_library(device_conv2d_fwd_instance ${DEVICE_CONV2D_FWD_INSTANCES})
