// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3r1.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DsLayout   = ck::Tuple<>;
using DsDataType = ck::Tuple<>;

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_FP16
void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_INT8))
void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);

void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
void add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif
#endif

#ifdef CK_USE_WMMA
#if defined(CK_ENABLE_FP16)
void add_device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_INT8))
void add_device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif

#if defined(CK_ENABLE_BF16)
void add_device_gemm_wmma_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               BF16,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances);
#endif
#endif

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmV2R1<ALayout,
                                                 BLayout,
                                                 DsLayout,
                                                 CLayout,
                                                 ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 CDataType,
                                                 ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmV2R1<ALayout,
                                    BLayout,
                                    DsLayout,
                                    CLayout,
                                    ADataType,
                                    BDataType,
                                    DsDataType,
                                    CDataType,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);

                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
#endif

#ifdef CK_USE_WMMA
                add_device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
#endif
            }
        }
#endif

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_INT8))
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_i8_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
#endif

#ifdef CK_USE_WMMA
                add_device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
#endif
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_reduce_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
#endif

#ifdef CK_USE_WMMA
                add_device_gemm_wmma_universal_reduce_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
#endif
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
