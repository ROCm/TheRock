// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_limits.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tuning_params.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_block_transfer.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_thread_block.hpp"

namespace ck_tile::builder::factory {

// Factory for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFwdXdlV3Factory
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts = internal::ConvTensorLayouts<SIGNATURE, SPATIAL_DIM, ConvDirection::FORWARD>;
    using Types   = internal::FwdConvTensorDataTypes<SIGNATURE>;
    using Ops     = internal::ElementwiseOps<SIGNATURE>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(ALGORITHM.transfer.a.lds_transfer.is_direct_load ==
                      ALGORITHM.transfer.b.lds_transfer.is_direct_load,
                  "A and B block transfers must both be direct load or not.");

    static constexpr bool IS_DIRECT_LOAD = ALGORITHM.transfer.a.lds_transfer.is_direct_load;
    static constexpr auto FWD_CONV_SPECIALIZATION = internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION     = internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                       .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto BLOCK         = internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.a>();
    static constexpr auto B_BLOCK_TRANSFER =
        internal::SetFwdConvBlockTransfer<ALGORITHM.transfer.b>();
    static constexpr auto C_BLOCK_TRANSFER = internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto BLOCK_GEMM       = internal::SetBlockGemm<ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        GRIDWISE_GEMM.m_per_xdl,
        GRIDWISE_GEMM.n_per_xdl,
        GRIDWISE_GEMM.m_xdl_per_wave,
        GRIDWISE_GEMM.n_xdl_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(A_BLOCK_TRANSFER.lds_padding),
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        static_cast<ck::index_t>(B_BLOCK_TRANSFER.lds_padding),
        C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        BLOCK_GEMM.scheduler,
        BLOCK_GEMM.pipeline_version,
        typename Types::AComputeType,
        typename Types::BComputeType,
        IS_DIRECT_LOAD>;
};

} // namespace ck_tile::builder::factory
