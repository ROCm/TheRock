// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))

using GemmF8F8BF16InstanceVector = std::vector<std::unique_ptr<
    DeviceGemmV2BPreshuffle<Row, Col, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&;

using GemmF8F8F16InstanceVector = std::vector<std::unique_ptr<
    DeviceGemmV2BPreshuffle<Row, Col, Row, F8, F8, F16, PassThrough, PassThrough, PassThrough>>>&;

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma32x32_mn_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma32x32_mn_compute_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p1_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p2_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p3_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p4_instances(
    GemmF8F8BF16InstanceVector& instances);
void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p5_instances(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_compute_instances_p1(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_compute_instances_p2(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part1(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part2(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part3(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part4(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part5(
    GemmF8F8BF16InstanceVector& instances);

void add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part6(
    GemmF8F8BF16InstanceVector& instances);

#endif
#if(defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8))
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p1(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p2(
    GemmF8F8F16InstanceVector& instances);

void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances_v2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances_v2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances_v2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances_v2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances_v2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p1(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p2(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p3(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p4(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p5(
    GemmF8F8F16InstanceVector& instances);
void add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p6(
    GemmF8F8F16InstanceVector& instances);
#endif
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
