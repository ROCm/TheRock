// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_blockscale_bpreshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
#if(defined(CK_USE_WMMA) && defined(CK_USE_WMMA_FP8))
void add_device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<Row,
                                                                                 Col,
                                                                                 Tuple<>,
                                                                                 Row,
                                                                                 F8,
                                                                                 F32,
                                                                                 F8,
                                                                                 F32,
                                                                                 Tuple<>,
                                                                                 BF16,
                                                                                 1,
                                                                                 128,
                                                                                 128,
                                                                                 PassThrough,
                                                                                 PassThrough,
                                                                                 PassThrough>>>&
        instances);

void add_device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<Row,
                                                                                 Col,
                                                                                 Tuple<>,
                                                                                 Row,
                                                                                 F8,
                                                                                 F32,
                                                                                 F8,
                                                                                 F32,
                                                                                 Tuple<>,
                                                                                 BF16,
                                                                                 1,
                                                                                 128,
                                                                                 128,
                                                                                 PassThrough,
                                                                                 PassThrough,
                                                                                 PassThrough>>>&
        instances);
#endif // CK_USE_WMMA && CK_USE_WMMA_FP8

#ifdef CK_USE_XDL
void add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffle<Row,
                                                                           Col,
                                                                           Tuple<>,
                                                                           Row,
                                                                           F8,
                                                                           F32,
                                                                           F8,
                                                                           F32,
                                                                           Tuple<>,
                                                                           BF16,
                                                                           1,
                                                                           128,
                                                                           128,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>>>&
        instances);

void add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffle<Row,
                                                                           Col,
                                                                           Tuple<>,
                                                                           Row,
                                                                           F8,
                                                                           F32,
                                                                           F8,
                                                                           F32,
                                                                           Tuple<>,
                                                                           BF16,
                                                                           1,
                                                                           128,
                                                                           128,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>>>&
        instances);

void add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffle<Row,
                                                                           Col,
                                                                           Tuple<>,
                                                                           Row,
                                                                           F8,
                                                                           F32,
                                                                           F8,
                                                                           F32,
                                                                           Tuple<>,
                                                                           BF16,
                                                                           1,
                                                                           128,
                                                                           128,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>>>&
        instances);

void add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_BlockScale_BPreshuffle<Row,
                                                                           Col,
                                                                           Tuple<>,
                                                                           Row,
                                                                           F8,
                                                                           F32,
                                                                           F8,
                                                                           F32,
                                                                           Tuple<>,
                                                                           BF16,
                                                                           1,
                                                                           128,
                                                                           128,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>>>&
        instances);
#endif
#endif

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<
        ALayout,
        BLayout,
        Tuple<>,
        CLayout,
        A0DataType,
        A1DataType,
        B0DataType,
        B1DataType,
        Tuple<>,
        CDataType,
        1,
        128,
        128,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<
        ALayout,
        BLayout,
        Tuple<>,
        CLayout,
        A0DataType,
        A1DataType,
        B0DataType,
        B1DataType,
        Tuple<>,
        CDataType,
        1,
        128,
        128,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if defined(CK_USE_XDL)
        // No XDL instances for DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK at the moment
#endif // CK_USE_XDL

#if(defined(CK_USE_WMMA) && defined(CK_USE_WMMA_FP8))
#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<A0DataType, f8_t> && is_same_v<B0DataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
                    op_ptrs);

                add_device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_1_128_128_mem_default_instances(
                    op_ptrs);
            }
        }
#endif
#endif // CK_USE_WMMA && CK_USE_WMMA_FP8

        return op_ptrs;
    }
};

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD_BlockScale_BPreshuffle<
        ALayout,
        BLayout,
        Tuple<>,
        CLayout,
        A0DataType,
        A1DataType,
        B0DataType,
        B1DataType,
        Tuple<>,
        CDataType,
        1,
        128,
        128,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp =
        DeviceGemmMultipleD_BlockScale_BPreshuffle<ALayout,
                                                   BLayout,
                                                   Tuple<>,
                                                   CLayout,
                                                   A0DataType,
                                                   A1DataType,
                                                   B0DataType,
                                                   B1DataType,
                                                   Tuple<>,
                                                   CDataType,
                                                   1,
                                                   128,
                                                   128,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<A0DataType, f8_t> && is_same_v<B0DataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_default_instances(
                    op_ptrs);
                // add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_comp_kpadding_instances(
                // op_ptrs);

                add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances(
                    op_ptrs);
                // add_device_gemm_blockscale_wp_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_kpadding_instances(
                //     op_ptrs);
            }
        }
#endif
#endif

#if defined(CK_USE_WMMA)
        // Reuse DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK instances
        using Wrapper = DeviceGemmMultipleD_BlockScale_BPreshuffleWrapper<
            ALayout,
            BLayout,
            Tuple<>,
            CLayout,
            A0DataType,
            A1DataType,
            B0DataType,
            B1DataType,
            Tuple<>,
            CDataType,
            1,
            128,
            128,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough>;
        auto new_op_ptrs =
            DeviceOperationInstanceFactory<typename Wrapper::DeviceOp>::GetInstances();
        for(auto& op_ptr : new_op_ptrs)
        {
            op_ptrs.emplace_back(std::make_unique<Wrapper>(std::move(op_ptr)));
        }
#endif // CK_USE_WMMA

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
