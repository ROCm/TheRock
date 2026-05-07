# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_POOL2D_FWD_INSTANCES)
list(APPEND DEVICE_POOL2D_FWD_INSTANCES device_avg_pool2d_fwd_nhwc_f16_instance.cpp
                                        device_max_pool2d_fwd_nhwc_f16_instance.cpp
                                        device_avg_pool2d_fwd_nhwc_f32_instance.cpp
                                        device_max_pool2d_fwd_nhwc_f32_instance.cpp
                                        device_avg_pool2d_fwd_nhwc_bf16_instance.cpp
                                        device_max_pool2d_fwd_nhwc_bf16_instance.cpp
                                        device_avg_pool2d_fwd_nhwc_i8_instance.cpp
                                        device_max_pool2d_fwd_nhwc_i8_instance.cpp
                                        device_avg_pool2d_fwd_nhwc_f8_instance.cpp
                                        device_max_pool2d_fwd_nhwc_f8_instance.cpp)
add_instance_library(device_pool2d_fwd_instance ${DEVICE_POOL2D_FWD_INSTANCES})
