# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_instance_library(device_softmax_instance
    device_softmax_f16_f16_instance_rank3_reduce1.cpp
    device_softmax_f16_f16_instance_rank3_reduce2.cpp
    device_softmax_f16_f16_instance_rank3_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce1.cpp
    device_softmax_f16_f16_instance_rank4_reduce2.cpp
    device_softmax_f16_f16_instance_rank4_reduce3.cpp
    device_softmax_f16_f16_instance_rank4_reduce4.cpp
    device_softmax_f32_f32_instance_rank3_reduce1.cpp
    device_softmax_f32_f32_instance_rank3_reduce2.cpp
    device_softmax_f32_f32_instance_rank3_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce1.cpp
    device_softmax_f32_f32_instance_rank4_reduce2.cpp
    device_softmax_f32_f32_instance_rank4_reduce3.cpp
    device_softmax_f32_f32_instance_rank4_reduce4.cpp)
