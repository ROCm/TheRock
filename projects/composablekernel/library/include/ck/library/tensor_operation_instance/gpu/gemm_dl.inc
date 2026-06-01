// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#if defined(CK_ENABLE_FP16)
void add_device_gemm_dl_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_km_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dpp_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dpp_f16_f16_f16_mk_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#if defined(CK_ENABLE_FP32)
void add_device_gemm_dl_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#if defined(CK_ENABLE_INT8)
void add_device_gemm_dl_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_km_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_km_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
