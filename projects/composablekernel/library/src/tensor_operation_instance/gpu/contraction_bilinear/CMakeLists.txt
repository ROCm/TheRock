# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(DEVICE_CONTRACTION_BILINEAR_INSTANCES)

list(APPEND DIMS 2 6)

foreach(idx IN LISTS DIMS)
    set(PREFIX ${idx}D/device_contraction_bilinear_m${idx}_n${idx}_k${idx})

    # FP32
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mnnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mnnn_instance.cpp)

    # FP64
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mnnn_instance.cpp)

    # FP16
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mnnn_instance.cpp)

    # BF16
    list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_kknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_knnn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mknn_instance.cpp
                                                        ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mnnn_instance.cpp)
endforeach()

add_instance_library(device_contraction_bilinear_instance ${DEVICE_CONTRACTION_BILINEAR_INSTANCES})
