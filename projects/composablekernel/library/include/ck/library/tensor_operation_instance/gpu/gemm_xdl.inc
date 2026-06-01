// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_INT8
void add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP64
void add_device_gemm_xdl_f64_f64_f64_km_kn_mn_instances(
    std::vector<std::unique_ptr<

        DeviceGemm<Col, Row, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP8
void add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_interwave_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v2_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_padded_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_interwave_padded_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v2_padded_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);

void add_device_gemm_xdl_c_shuffle_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f8_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
