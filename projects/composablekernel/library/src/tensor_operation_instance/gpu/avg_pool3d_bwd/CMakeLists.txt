# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_AVGPOOL_BWD_INSTANCES)
list(APPEND DEVICE_AVGPOOL_BWD_INSTANCES device_avg_pool3d_bwd_ndhwc_f16_instance.cpp
                                         device_avg_pool3d_bwd_ndhwc_bf16_instance.cpp
                                         device_avg_pool3d_bwd_ndhwc_f32_instance.cpp)
add_instance_library(device_avg_pool3d_bwd_instance ${DEVICE_AVGPOOL_BWD_INSTANCES})
