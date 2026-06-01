# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_NORMALIZATION_bwd_data_INSTANCES)

list(APPEND DEVICE_NORMALIZATION_bwd_data_INSTANCES
    device_groupnorm_bwd_data_f32_instance.cpp
    device_layernorm2d_bwd_data_f16_instance.cpp
    device_layernorm2d_bwd_data_f32_instance.cpp)

add_instance_library(device_normalization_bwd_data_instance ${DEVICE_NORMALIZATION_bwd_data_INSTANCES})
