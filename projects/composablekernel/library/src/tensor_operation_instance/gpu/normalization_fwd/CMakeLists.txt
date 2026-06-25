# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_NORMALIZATION_FWD_INSTANCES)

list(APPEND DEVICE_NORMALIZATION_FWD_INSTANCES
    device_layernorm2d_fwd_f16_instance.cpp
    device_layernorm4d_fwd_f16_instance.cpp
    device_groupnorm_fwd_f16_instance.cpp
    device_groupnorm_fwd_swish_f16_instance.cpp
    device_groupnorm_fwd_swish_f16_f32_f32_f16_instance.cpp
    device_layernorm2d_fwd_f32_instance.cpp
    device_layernorm4d_fwd_f32_instance.cpp
    device_groupnorm_fwd_f32_instance.cpp
    device_groupnorm_fwd_swish_f32_instance.cpp)

add_instance_library(device_normalization_fwd_instance ${DEVICE_NORMALIZATION_FWD_INSTANCES})
