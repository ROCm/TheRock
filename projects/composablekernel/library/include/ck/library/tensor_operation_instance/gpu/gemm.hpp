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

#ifdef DL_KERNELS
#include "gemm_dl.inc"
#endif
#ifdef DPP_KERNELS
#include "gemm_dpp.inc"
#endif
#ifdef CK_USE_WMMA
#include "gemm_wmma.inc"
#endif
#ifdef CK_USE_XDL
#include "gemm_xdl.inc"
#endif

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemm<ALayout,
                                             BLayout,
                                             CLayout,
                                             ADataType,
                                             BDataType,
                                             CDataType,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemm<ALayout,
                                BLayout,
                                CLayout,
                                ADataType,
                                BDataType,
                                CDataType,
                                ck::tensor_operation::element_wise::PassThrough,
                                ck::tensor_operation::element_wise::PassThrough,
                                ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef DL_KERNELS
        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<CDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f32_f32_f32_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f32_f32_f32_km_nk_mn_instances(op_ptrs);
            }
        }
#ifdef CK_ENABLE_FP16
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_mk_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_mk_nk_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f16_f16_f16_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_f16_f16_f16_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_km_nk_mn_irregular_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_INT8
        else if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                          is_same_v<CDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_i8_i8_i8_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_km_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dl_i8_i8_i8_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_km_nk_mn_irregular_instances(op_ptrs);
            }
        }
#endif
#endif // DL_KERNELS

#ifdef DPP_KERNELS
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_dpp_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dpp_f16_f16_f16_mk_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dpp_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dpp_f16_f16_f16_mk_nk_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dpp_f16_f16_f16_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_dpp_f16_f16_f16_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dpp_f16_f16_f16_km_nk_mn_irregular_instances(op_ptrs);
            }
        }
#endif
#endif // DPP_KERNELS

#ifdef CK_USE_WMMA
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> &&
                     is_same_v<CDataType, ck::bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_bf16_bf16_bf16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_bf16_bf16_bf16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_int8_int8_int8_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_int8_int8_int8_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_int8_int8_int8_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_wmma_int8_int8_int8_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#endif

#ifdef CK_USE_XDL
        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<CDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_kn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_mk_nk_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_kn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_lds_direct_load_f32_f32_f32_km_nk_mn_instances(
                    op_ptrs);
            }
        }
#ifdef CK_ENABLE_FP16
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_lds_direct_load_f16_f16_f16_mk_nk_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_BF16
        else if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> &&
                          is_same_v<CDataType, ck::bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_INT8
        else if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                          is_same_v<CDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP8
        else if constexpr(is_same_v<ADataType, ck::f8_t> && is_same_v<BDataType, ck::f8_t> &&
                          is_same_v<CDataType, ck::f8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_padded_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_interwave_padded_instances(
                    op_ptrs);
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v2_padded_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_default_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_interwave_default_instances(
                    op_ptrs);
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v2_default_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, ck::half_t> && is_same_v<BDataType, ck::f8_t> &&
                          is_same_v<CDataType, ck::half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f8_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_f16_f8_f16_mk_nk_mn_instances(op_ptrs);
            }
        }
#endif
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
