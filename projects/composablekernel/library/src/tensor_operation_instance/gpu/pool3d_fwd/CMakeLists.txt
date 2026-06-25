# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_POOL3D_FWD_INSTANCES)
list(APPEND DEVICE_POOL3D_FWD_INSTANCES device_avg_pool3d_fwd_ndhwc_f16_instance.cpp
                                        device_max_pool3d_fwd_ndhwc_f16_instance.cpp
                                        device_max_pool3d_fwd_ndhwc_f8_instance.cpp
                                        device_avg_pool3d_fwd_ndhwc_f8_instance.cpp
                                        device_max_pool3d_fwd_ndhwc_i8_instance.cpp
                                        device_avg_pool3d_fwd_ndhwc_i8_instance.cpp
                                        device_avg_pool3d_fwd_ndhwc_f32_instance.cpp
                                        device_max_pool3d_fwd_ndhwc_f32_instance.cpp
                                        device_avg_pool3d_fwd_ndhwc_bf16_instance.cpp
                                        device_max_pool3d_fwd_ndhwc_bf16_instance.cpp)
add_instance_library(device_pool3d_fwd_instance ${DEVICE_POOL3D_FWD_INSTANCES})
