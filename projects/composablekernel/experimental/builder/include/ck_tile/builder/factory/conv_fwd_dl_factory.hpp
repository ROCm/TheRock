// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tuning_params.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_thread_block.hpp"

namespace ck_tile::builder::factory {

// Factory for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFwdDlFactory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
    using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops     = internal::ElementwiseOps<SIGNATURE>;
    using AlgorithmType = decltype(ALGORITHM);

    static constexpr auto FWD_CONV_SPECIALIZATION = internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION     = internal::SetGemmSpecialization<ALGORITHM>();

    static constexpr auto BLOCK = internal::SetThreadBlockInfo<ALGORITHM>();

    // DL-specific parameters from algorithm descriptor
    static constexpr auto DL_THREAD_CFG      = ALGORITHM.thread_config;
    static constexpr ck::index_t K0PerBlock  = DL_THREAD_CFG.k0_per_block;
    static constexpr ck::index_t K1          = DL_THREAD_CFG.k1;
    static constexpr ck::index_t M1PerThread = DL_THREAD_CFG.m1_per_thread;
    static constexpr ck::index_t N1PerThread = DL_THREAD_CFG.n1_per_thread;
    static constexpr ck::index_t KPerThread  = DL_THREAD_CFG.k_per_thread;

    // Thread cluster from descriptor
    static constexpr auto DL_CLUSTER = ALGORITHM.thread_cluster;
    using M1N1ThreadClusterM1Xs      = to_sequence_v<DL_CLUSTER.m1_xs>;
    using M1N1ThreadClusterN1Xs      = to_sequence_v<DL_CLUSTER.n1_xs>;

    // A Block Transfer from descriptor - K0_M0_M1_K1 tensor format
    static constexpr auto DL_A_TRANSFER = ALGORITHM.transfer.a.block_transfer;
    using ABlockTransferThreadSliceLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_slice_lengths>;
    using ABlockTransferThreadClusterLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_lengths>;
    using ABlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_arrange_order>;
    using ABlockTransferSrcAccessOrder = to_sequence_v<DL_A_TRANSFER.src_access_order>;
    using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_lengths>;
    using ABlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.dst_vector_tensor_lengths>;

    // B Block Transfer from descriptor - K0_N0_N1_K1 tensor format
    static constexpr auto DL_B_TRANSFER = ALGORITHM.transfer.b.block_transfer;
    using BBlockTransferThreadSliceLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_slice_lengths>;
    using BBlockTransferThreadClusterLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_lengths>;
    using BBlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_arrange_order>;
    using BBlockTransferSrcAccessOrder = to_sequence_v<DL_B_TRANSFER.src_access_order>;
    using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_lengths>;
    using BBlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.dst_vector_tensor_lengths>;

    // C Thread Transfer from descriptor
    static constexpr auto DL_C_TRANSFER    = ALGORITHM.transfer.c.epilogue;
    using CThreadTransferSrcDstAccessOrder = to_sequence_v<DL_C_TRANSFER.src_dst_access_order>;
    static constexpr ck::index_t CThreadTransferSrcDstVectorDim = DL_C_TRANSFER.src_dst_vector_dim;
    static constexpr ck::index_t CThreadTransferDstScalarPerVector =
        DL_C_TRANSFER.dst_scalar_per_vector;

    // The DL forward convolution kernel class instance
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
        SPATIAL_DIM,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Types::AccDataType,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        FWD_CONV_SPECIALIZATION,
        GEMM_SPECIALIZATION,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        K0PerBlock,
        K1,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM1Xs,
        M1N1ThreadClusterN1Xs,
        ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
        ABlockTransferSrcVectorTensorContiguousDimOrder,
        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
        BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
        BBlockTransferSrcVectorTensorContiguousDimOrder,
        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;
};

} // namespace ck_tile::builder::factory
