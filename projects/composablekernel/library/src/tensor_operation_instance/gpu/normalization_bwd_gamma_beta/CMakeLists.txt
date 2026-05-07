# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(DEVICE_NORMALIZATION_BWD_GAMMA_BETA_INSTANCES)

list(APPEND DEVICE_NORMALIZATION_BWD_GAMMA_BETA_INSTANCES
    device_groupnorm_bwd_gamma_beta_f32_instance.cpp
    device_layernorm2d_bwd_gamma_beta_f16_instance.cpp
    device_layernorm2d_bwd_gamma_beta_f32_instance.cpp)

add_instance_library(device_normalization_bwd_gamma_beta_instance ${DEVICE_NORMALIZATION_BWD_GAMMA_BETA_INSTANCES})
