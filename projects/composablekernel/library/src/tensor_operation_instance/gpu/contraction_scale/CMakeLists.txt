# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(DEVICE_CONTRACTION_SCALE_INSTANCES)

list(APPEND DIMS 2 6)

foreach(idx IN LISTS DIMS)
    set(PREFIX ${idx}D/device_contraction_scale_m${idx}_n${idx}_k${idx})

    # FP32
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance.cpp)

    # FP64
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f64_f64_f64_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_mnn_instance.cpp)

    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance.cpp)

    # FP16
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance.cpp)

    # BF16
    list(APPEND DEVICE_CONTRACTION_SCALE_INSTANCES ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance.cpp
                                                    ${PREFIX}_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance.cpp)
endforeach()

add_instance_library(device_contraction_scale_instance ${DEVICE_CONTRACTION_SCALE_INSTANCES})

