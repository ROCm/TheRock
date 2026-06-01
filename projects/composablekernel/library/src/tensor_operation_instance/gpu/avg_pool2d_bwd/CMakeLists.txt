# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_AVGPOOL_2D_BWD_INSTANCES)
list(APPEND DEVICE_AVGPOOL_2D_BWD_INSTANCES device_avg_pool2d_bwd_nhwc_bf16_instance.cpp
                                            device_avg_pool2d_bwd_nhwc_f16_instance.cpp
                                            device_avg_pool2d_bwd_nhwc_f32_instance.cpp
                                            device_avg_pool2d_bwd_nhwc_f8_instance.cpp
                                            device_avg_pool2d_bwd_nhwc_int8_instance.cpp)
add_instance_library(device_avg_pool2d_bwd_instance ${DEVICE_AVGPOOL_2D_BWD_INSTANCES})

