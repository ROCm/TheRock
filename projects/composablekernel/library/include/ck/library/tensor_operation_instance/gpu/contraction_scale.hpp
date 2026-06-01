// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// 2D
#ifdef CK_ENABLE_FP32
void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);
#endif // CK_ENABLE_FP32

#ifdef CK_ENABLE_FP64
void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP64

#ifdef CK_ENABLE_FP16
void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP16

#ifdef CK_ENABLE_BF16
void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP16
// 6D
#ifdef CK_ENABLE_FP32
void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           BF16>>>& instances);
#endif // CK_ENABLE_FP32

#ifdef CK_ENABLE_FP64
void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F64>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F64,
                                                           F64,
                                                           Empty_Tuple,
                                                           F64,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP64

#ifdef CK_ENABLE_FP16
void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           F16,
                                                           F16,
                                                           Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP16

#ifdef CK_ENABLE_BF16
void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);

void add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                           6,
                                                           6,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances);
#endif // CK_ENABLE_FP16

// Contraction + Scale
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename ComputeDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceContractionMultipleD<
    NumDimM,
    NumDimN,
    NumDimK,
    ADataType,
    BDataType,
    ck::Tuple<>,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale,
    ComputeDataType>>
{
    using DeviceOp = DeviceContractionMultipleD<NumDimM,
                                                NumDimN,
                                                NumDimK,
                                                ADataType,
                                                BDataType,
                                                ck::Tuple<>,
                                                EDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::Scale,
                                                ComputeDataType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<EDataType, float>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, ck::half_t>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, ck::bhalf_t>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance(
                        op_ptrs);
                }
            }
            else if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, ck::half_t>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, ck::bhalf_t>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance(
                        op_ptrs);
                }
            }
        }
#endif // CK_ENABLE_FP32
#ifdef CK_ENABLE_FP64
        if constexpr(is_same_v<ADataType, double> && is_same_v<BDataType, double> &&
                     is_same_v<EDataType, double>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                if constexpr(is_same_v<ComputeDataType, double>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
            else if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
            {
                if constexpr(is_same_v<ComputeDataType, double>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mnn_instance(
                        op_ptrs);
                }
                else if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
        }
#endif // CK_ENABLE_FP64
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, ck::half_t> && is_same_v<BDataType, ck::half_t> &&
                     is_same_v<EDataType, ck::half_t>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
            else if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
        }
#endif // CK_ENABLE_FP16
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> &&
                     is_same_v<EDataType, ck::bhalf_t>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
            else if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
            {
                if constexpr(is_same_v<ComputeDataType, float>)
                {
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance(
                        op_ptrs);
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance(
                        op_ptrs);
                }
            }
        }
#endif // CK_ENABLE_BF16
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
