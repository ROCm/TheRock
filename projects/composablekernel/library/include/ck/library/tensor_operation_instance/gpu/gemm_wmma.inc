// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_wmma_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_int8_int8_int8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_int8_int8_int8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_int8_int8_int8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_wmma_int8_int8_int8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
