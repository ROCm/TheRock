// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_conv_v3.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_offset_utils.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          index_t NumGroupsToMerge,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage(
        typename GridwiseGemm::Argument karg,
        [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        [[maybe_unused]] const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        [[maybe_unused]] const index_t num_k_per_block,
        const long_index_t split_k_stride_a,
        const long_index_t split_k_stride_b,
        bool split_k_offset_hack)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z * NumGroupsToMerge);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const long_index_t split_k_offset_a = split_k_offset_hack ? k_idx * split_k_stride_a : 0;
        const long_index_t split_k_offset_b = split_k_offset_hack ? k_idx * split_k_stride_b : 0;

        const long_index_t a_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
        const long_index_t b_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
        const long_index_t e_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        DispatchSplitKHack<GridwiseGemm,
                           AGridDesc_AK0_M_K1,
                           BGridDesc_BK0_N_K1,
                           CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                           HasMainKBlockLoop,
                           CGlobalMemoryDataOperation,
                           TailNum>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                    karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                    karg.p_c_grid + e_batch_offset,
                                    p_shared,
                                    karg,
                                    a_grid_desc_ak0_m_ak1,
                                    b_grid_desc_bk0_n_bk1,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    k_idx * num_k_per_block,
                                    gridDim.y,
                                    split_k_offset_hack);
    }
#else
    ignore = karg;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
    ignore = split_k_offset_hack;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          index_t NumGroupsToMerge,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds(
        typename GridwiseGemm::Argument karg,
        [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        [[maybe_unused]] const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        [[maybe_unused]] const index_t num_k_per_block,
        const long_index_t split_k_stride_a,
        const long_index_t split_k_stride_b,
        bool split_k_offset_hack)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // offset base pointer for each work-group
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z * NumGroupsToMerge);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const long_index_t split_k_offset_a = split_k_offset_hack ? k_idx * split_k_stride_a : 0;
        const long_index_t split_k_offset_b = split_k_offset_hack ? k_idx * split_k_stride_b : 0;

        const long_index_t a_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
        const long_index_t b_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
        const long_index_t e_batch_offset = amd_wave_read_first_lane(
            static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        DispatchSplitKHack_2Lds<GridwiseGemm,
                                AGridDesc_AK0_M_K1,
                                BGridDesc_BK0_N_K1,
                                CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                HasMainKBlockLoop,
                                CGlobalMemoryDataOperation,
                                TailNum>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                         karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                         karg.p_c_grid + e_batch_offset,
                                         p_shared_0,
                                         p_shared_1,
                                         karg,
                                         a_grid_desc_ak0_m_ak1,
                                         b_grid_desc_bk0_n_bk1,
                                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                                         k_idx * num_k_per_block,
                                         gridDim.y,
                                         split_k_offset_hack);
    }
#else
    ignore = karg;
    ignore = split_k_offset_hack;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
#endif // end of if (defined(__gfx9__))
}

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          index_t NumGroupsToMerge                    = 1,
          typename ComputeTypeA                       = InDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          index_t TransposeTransferSrcScalarPerVector = 1,
          index_t TransposeTransferDstScalarPerVector = 1>
struct DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        ComputeTypeA,
                                        ComputeTypeB>
{
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    using DeviceOp = DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using EDataType = WeiDataType;

    // If NGCHW then ADataType must be equal to BDataType
    static_assert(!(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                    is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>()) ||
                  is_same_v<ADataType, BDataType>);

    using AElementwiseOperation   = OutElementwiseOperation;
    using BElementwiseOperation   = InElementwiseOperation;
    using CDEElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto conv_to_gemm_transformer_v2 =
        TransformConvBwdWeightToGemmV2<NDimSpatial,
                                       MPerBlock,
                                       NPerBlock,
                                       K1Number,
                                       KPerBlock / K1Number,
                                       NumGroupsToMerge,
                                       ConvBackwardWeightSpecialization>{};

    static constexpr auto conv_to_gemm_transformer_v1 =
        TransformConvBwdWeightToGemm<NDimSpatial,
                                     MPerBlock,
                                     NPerBlock,
                                     K1Number,
                                     KPerBlock / K1Number,
                                     ConvBackwardWeightSpecialization>{};

    static constexpr index_t ClusterLengthMPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto conv_ngchw_to_nhwgc_transformer =
        TransformConvNGCHWToNHWGC<InLayout,
                                  WeiLayout,
                                  OutLayout,
                                  NDimSpatial,
                                  MPerBlock / ClusterLengthMPerBlock,
                                  NPerBlock / ClusterLengthNPerBlock>{};

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer_v2
            .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(dim,
                                                                         dim,
                                                                         dim,
                                                                         lengths,
                                                                         lengths,
                                                                         lengths,
                                                                         strides,
                                                                         strides,
                                                                         strides,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer_v2
            .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(dim,
                                                                         dim,
                                                                         dim,
                                                                         lengths,
                                                                         lengths,
                                                                         lengths,
                                                                         strides,
                                                                         strides,
                                                                         strides,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetElementwiseCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer_v1
            .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(dim,
                                                                         dim,
                                                                         dim,
                                                                         lengths,
                                                                         lengths,
                                                                         lengths,
                                                                         strides,
                                                                         strides,
                                                                         strides,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         batch)[I2];
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetElementwiseCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer_v1
            .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(dim,
                                                                         dim,
                                                                         dim,
                                                                         lengths,
                                                                         lengths,
                                                                         lengths,
                                                                         strides,
                                                                         strides,
                                                                         strides,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         params,
                                                                         batch)[I2];
    }

    using NGCHWTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNGCHWTransposeDesc<NDimSpatial>({}, {}))>;
    using NHWGCTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNHWGCTransposeDesc<NDimSpatial>({}, {}))>;
    using GKCYXTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeGKCYXTransposeDesc<NDimSpatial>({}, {}))>;
    using GKYXCTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeGKYXCTransposeDesc<NDimSpatial>({}, {}))>;

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;
    using CElementwiseGridDesc_M_N =
        remove_cvref_t<decltype(GetElementwiseCGridDesc<NDimSpatial>())>;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_xdl_cshuffle_conv_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        AccDataType,
        AccDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        K1,
        K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    using GridwiseElementwiseCast =
        GridwiseElementwise<Tuple<CElementwiseGridDesc_M_N>,
                            Tuple<CElementwiseGridDesc_M_N>,
                            Tuple<const AccDataType*>,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            CDEElementwiseOperation,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<0, 1>,
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            I1,
                            I1>;
    // NPerBlock is used for the first dim which is store dimension
    // (with CBlockTransferScalarPerVector_NWaveNPerXdl scalar per vector).
    // CBlockTransferScalarPerVector_NWaveNPerXdl is aligned to NPerBlock so
    // it is more flexible to use this dim for store dimension with such scalar
    // per vector.
    using GridwiseElementwiseWeightTransposeCast =
        GridwiseElementwise<Tuple<GKYXCTransposeDescType>,
                            Tuple<GKCYXTransposeDescType>,
                            Tuple<const AccDataType*>,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            CDEElementwiseOperation,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<0, 1>,
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            Sequence<1>,
                            I1,
                            I0>;

    using GridwiseElementwiseTranspose =
        GridwiseElementwise<Tuple<NGCHWTransposeDescType>,
                            Tuple<NHWGCTransposeDescType>,
                            Tuple<const ADataType*>,
                            Tuple<ADataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<TransposeTransferSrcScalarPerVector>,
                            Sequence<TransposeTransferDstScalarPerVector>,
                            I1,
                            I0>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}, 1, 1));

    struct ActiveWorkgroupsPerCU
    {
        template <typename GridwiseGemm>
        int GetMaxOccupancy()
        {
            constexpr int dynamic_smem_size = 0;
            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;
            int max_occupancy = 0;

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
            {
                hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_occupancy,
                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds<
                        GridwiseGemm,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        NumGroupsToMerge,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>,
                    BlockSize,
                    dynamic_smem_size));
            }
            else
            {
                hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_occupancy,
                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                        GridwiseGemm,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        NumGroupsToMerge,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>,
                    BlockSize,
                    dynamic_smem_size));
            }
            return std::max(1, max_occupancy);
        }

        ActiveWorkgroupsPerCU()
        {
            max_occupancy_ = 1;
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm64>();
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm32>();
                }
            }
        }
        int max_occupancy_;
    };

    struct Argument : public BaseArgument, public ArgumentSplitK
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 const ck::index_t M01,
                 const ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_e_grid_{p_wei_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              ce_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              cde_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            static ActiveWorkgroupsPerCU active_workgroups_per_cu;

            c_space_size_bytes =
                ck::accumulate_n<long_index_t>(
                    e_g_k_c_xs_lengths.begin(), NDimSpatial + I3, 1, std::multiplies<>()) *
                sizeof(AccDataType);

            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_transposed =
                conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(a_g_n_k_wos_lengths,
                                                                      a_g_n_k_wos_strides);
            std::array<index_t, NDimSpatial + 3> b_g_n_c_wis_strides_transposed =
                conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(b_g_n_c_wis_lengths,
                                                                      b_g_n_c_wis_strides);
            std::array<index_t, NDimSpatial + 3> e_g_k_c_xs_strides_transposed =
                conv_ngchw_to_nhwgc_transformer.TransposeWeiStrides(e_g_k_c_xs_lengths,
                                                                    e_g_k_c_xs_strides);

            if(split_k < 0)
            {
                ck::index_t gemmM, gemmN, gemmK;
                std::tie(gemmM, gemmN, gemmK) =
                    get_bwd_weight_gemm_sizes<NDimSpatial>(a_g_n_k_wos_lengths, e_g_k_c_xs_lengths);

                const auto grid_size = calculate_mn_grid_size<MPerBlock, NPerBlock>(gemmM, gemmN) *
                                       Conv_G_ / NumGroupsToMerge;
                k_batch_ = get_best_occupancy_k_batch_value(active_workgroups_per_cu.max_occupancy_,
                                                            grid_size);

                // Ensure that k_batch_ does not exceed the maximum value
                // for the GEMM pipeline.
                const auto k_batch_max = static_cast<index_t>((gemmK - 1) / KPerBlock);
                k_batch_               = std::max(std::min(k_batch_, k_batch_max), 1);

                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[SPLIT-K AUTODEDUCE] k_batch max value: " << k_batch_max
                              << std::endl;
                    std::cout << "[SPLIT-K AUTODEDUCE] Final k_batch value: " << k_batch_
                              << std::endl;
                }
            }
            else
            {
                k_batch_ = split_k;
            }

            // Create initial descriptors with hack=false to check compactness
            const auto descs_initial =
                conv_to_gemm_transformer_v2
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides_transposed,
                        e_g_k_c_xs_strides_transposed,
                        a_g_n_k_wos_strides_transposed,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_,
                        false, // hack=false for initial check
                        true); // use_full_batch_kindex

            ce_elementwise_grid_desc_m_n_ =
                conv_to_gemm_transformer_v1
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_)[I2];

            split_k_offset_hack_ =
                SplitKHackEligibility<NDimSpatial, InLayout, WeiLayout, OutLayout>::Check(
                    descs_initial[I0],
                    descs_initial[I1],
                    k_batch_,
                    Conv_N_,
                    output_spatial_lengths_,
                    KPerBlock);

            // Create final descriptors with correct hack flag
            const auto descs =
                conv_to_gemm_transformer_v2
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides_transposed,
                        e_g_k_c_xs_strides_transposed,
                        a_g_n_k_wos_strides_transposed,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_,
                        split_k_offset_hack_, // Use determined hack flag
                        true);                // use_full_batch_kindex

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            ce_grid_desc_m_n_    = descs[I2];

            // Step 5: Calculate stride using CalculateOffset on FINAL descriptors
            if(split_k_offset_hack_)
            {
                const index_t k0_per_batch = a_grid_desc_k0_m_k1_.GetLength(I0) / k_batch_;
                const auto idx_start       = make_multi_index(0, 0, 0);
                const auto idx_next        = make_multi_index(k0_per_batch, 0, 0);
                split_k_stride_a_          = a_grid_desc_k0_m_k1_.CalculateOffset(idx_next) -
                                    a_grid_desc_k0_m_k1_.CalculateOffset(idx_start);
            }
            else
            {
                split_k_stride_a_ = a_grid_desc_k0_m_k1_.GetElementSpaceSize();
            }

            if(split_k_offset_hack_)
            {
                const index_t k0_per_batch = b_grid_desc_k0_n_k1_.GetLength(I0) / k_batch_;
                const auto idx_start       = make_multi_index(0, 0, 0);
                const auto idx_next        = make_multi_index(k0_per_batch, 0, 0);
                split_k_stride_b_          = b_grid_desc_k0_n_k1_.CalculateOffset(idx_next) -
                                    b_grid_desc_k0_n_k1_.CalculateOffset(idx_start);
            }
            else
            {
                split_k_stride_b_ = b_grid_desc_k0_n_k1_.GetElementSpaceSize();
            }

            const index_t GemmM = a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = b_grid_desc_k0_n_k1_.GetLength(I1);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides_transposed[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides_transposed[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ = e_g_k_c_xs_strides_transposed[0];
            c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ce_grid_desc_m_n_,
                    GridwiseGemm64::CalculateMBlock(GemmM),
                    GridwiseGemm64::CalculateNBlock(GemmN));

            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                a_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        a_g_n_k_wos_lengths, a_g_n_k_wos_strides);
                a_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        a_g_n_k_wos_lengths, a_g_n_k_wos_strides);

                b_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        b_g_n_c_wis_lengths, b_g_n_c_wis_strides);
                b_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        b_g_n_c_wis_lengths, b_g_n_c_wis_strides);

                e_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeGKYXCTransposeDesc<NDimSpatial>(
                        e_g_k_c_xs_lengths, e_g_k_c_xs_strides);
                e_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeGKCYXTransposeDesc<NDimSpatial>(
                        e_g_k_c_xs_lengths, e_g_k_c_xs_strides);

                elementwise_block_2_ctile_map_transpose_a_ = Block2TileMapElementwise{
                    a_in_transpose_desc_.GetLength(I0), a_in_transpose_desc_.GetLength(I1)};

                elementwise_block_2_ctile_map_transpose_b_ = Block2TileMapElementwise{
                    b_in_transpose_desc_.GetLength(I0), b_in_transpose_desc_.GetLength(I1)};
            }

            elementwise_block_2_ctile_map_ =
                is_NGCHW_GKCYX_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                        is_NGCDHW_GKCZYX_NGKDHW<InLayout, WeiLayout, OutLayout>()
                    ? Block2TileMapElementwise{e_in_transpose_desc_.GetLength(I0),
                                               e_in_transpose_desc_.GetLength(I1)}
                    : Block2TileMapElementwise{ce_grid_desc_m_n_.GetLength(I0),
                                               ce_grid_desc_m_n_.GetLength(I1)};
        }

        std::size_t GetWorkspaceATensorSizeBytes() const
        {
            // Align to 128B
            return math::integer_divide_ceil(
                       sizeof(ADataType) * a_in_transpose_desc_.GetElementSpaceSize(), 128) *
                   128;
        }

        std::size_t GetWorkspaceBTensorSizeBytes() const
        {
            return sizeof(BDataType) * b_in_transpose_desc_.GetElementSpaceSize();
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            // Align to 128B
            return math::integer_divide_ceil(sizeof(AccDataType) *
                                                 ce_grid_desc_m_n_.GetElementSpaceSize() * Conv_G_,
                                             128) *
                   128;
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            // 1. We need to transpose A and B for NGCHW and NGKHW layouts
            // 2. If C format is GKCYX then tranpose during second stage.
            //    If C format is GKYXC then just perform second stage.
            //    Due to the fact that E workspace is always needed, we
            //    allocate them as the first part of the workspace.
            //    [EWorkspace, AWorkspace, BWorkspace]
            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                return GetWorkspaceATensorSizeBytes() + GetWorkspaceBTensorSizeBytes() +
                       GetWorkspaceETensorSizeBytes();
            }
            else
            {
                return GetWorkspaceETensorSizeBytes();
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N ce_grid_desc_m_n_;
        CElementwiseGridDesc_M_N ce_elementwise_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;

        Block2TileMapElementwise elementwise_block_2_ctile_map_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_b_;

        NGCHWTransposeDescType a_in_transpose_desc_, b_in_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, b_out_transpose_desc_;
        GKYXCTransposeDescType e_in_transpose_desc_;
        GKCYXTransposeDescType e_out_transpose_desc_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        long_index_t c_space_size_bytes;

        bool split_k_offset_hack_;
        long_index_t split_k_stride_a_, split_k_stride_b_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                      << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                      << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.ce_grid_desc_m_n_{" << arg.ce_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.ce_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        template <typename GridwiseGemm>
        float RunGemmV3(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t GemmM = arg.a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_k0_n_k1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            AccDataType* p_c_grid = type_convert<AccDataType*>(arg.p_workspace_);

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;

            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                p_a_grid = type_convert<const ADataType*>(arg.p_workspace_) +
                           arg.GetWorkspaceETensorSizeBytes() / sizeof(ADataType);
                p_b_grid =
                    type_convert<const BDataType*>(arg.p_workspace_) +
                    (arg.GetWorkspaceETensorSizeBytes() + arg.GetWorkspaceATensorSizeBytes()) /
                        sizeof(BDataType);
            }

            // nullptr for output, will be set after workspace set
            typename GridwiseGemm::Argument gemm_arg{
                p_a_grid, p_b_grid, p_c_grid, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
                gemm_arg.M, gemm_arg.N, gemm_arg.KBatch, arg.Conv_G_ / NumGroupsToMerge);

            float ave_time = 0;

            index_t k_grain = gemm_arg.KBatch * KPerBlock;
            index_t K_split = (gemm_arg.K + k_grain - 1) / k_grain * (KPerBlock);

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto num_k_per_block =
                arg.a_grid_desc_k0_m_k1_.GetLength(Number<0>{}) / gemm_arg.KBatch;

            const auto clear_workspace = [&]() {
                if(arg.k_batch_ > 1)
                {
                    hip_check_error(hipMemsetAsync(
                        gemm_arg.p_c_grid, 0, arg.c_space_size_bytes, stream_config.stream_id_));
                }
            };

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        gemm_arg_,
                        stream_config.rotating_count,
                        gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                        gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        clear_workspace();
                    };

                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
                else
                {
                    ave_time += launch_and_time_kernel_with_preprocess(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            NumGroupsToMerge,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            NumGroupsToMerge,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::AtomicAdd,
                                        minimum_occupancy,
                                        TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel =
                                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                        GridwiseGemm,
                                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                        remove_reference_t<
                                            DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                        NumGroupsToMerge,
                                        true,
                                        InMemoryDataOperationEnum::Set,
                                        minimum_occupancy,
                                        TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage_2lds<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
                else
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel =
                                kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            NumGroupsToMerge,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_two_stage<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            NumGroupsToMerge,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time                 = 0.f;
            auto launch_elementwise_kernel = [&]() {
                const AccDataType* p_c_grid = type_convert<const AccDataType*>(arg.p_workspace_);

                std::array<index_t, I1> in_out_batch_strides = {
                    static_cast<index_t>(arg.compute_ptr_offset_of_batch_.BatchStrideC_)};

                if constexpr(is_NGCHW_GKCYX_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                             is_NGCDHW_GKCZYX_NGKDHW<InLayout, WeiLayout, OutLayout>())
                {
                    const index_t grid_size = arg.elementwise_block_2_ctile_map_.CalculateGridSize(
                        arg.e_in_transpose_desc_);

                    const auto kernel = kernel_elementwise<GridwiseElementwiseWeightTransposeCast,
                                                           ck::Tuple<GKYXCTransposeDescType>,
                                                           ck::Tuple<GKCYXTransposeDescType>,
                                                           ck::Tuple<const AccDataType*>,
                                                           ck::Tuple<EDataType*>,
                                                           Block2TileMapElementwise,
                                                           CDEElementwiseOperation>;

                    return launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  make_tuple(arg.e_in_transpose_desc_),
                                                  make_tuple(arg.e_out_transpose_desc_),
                                                  make_tuple(p_c_grid),
                                                  make_tuple(arg.p_e_grid_),
                                                  arg.elementwise_block_2_ctile_map_,
                                                  arg.cde_element_op_);
                }
                else
                {
                    const index_t grid_size = arg.elementwise_block_2_ctile_map_.CalculateGridSize(
                                                  arg.ce_elementwise_grid_desc_m_n_) *
                                              arg.Conv_G_;

                    const auto kernel =
                        kernel_batched_elementwise<GridwiseElementwiseCast,
                                                   ck::Tuple<CElementwiseGridDesc_M_N>,
                                                   ck::Tuple<CElementwiseGridDesc_M_N>,
                                                   ck::Tuple<const AccDataType*>,
                                                   ck::Tuple<EDataType*>,
                                                   Block2TileMapElementwise,
                                                   CDEElementwiseOperation,
                                                   I1,
                                                   I1>;

                    return launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  make_tuple(arg.ce_elementwise_grid_desc_m_n_),
                                                  make_tuple(arg.ce_elementwise_grid_desc_m_n_),
                                                  make_tuple(p_c_grid),
                                                  make_tuple(arg.p_e_grid_),
                                                  arg.elementwise_block_2_ctile_map_,
                                                  arg.cde_element_op_,
                                                  arg.Conv_G_,
                                                  in_out_batch_strides,
                                                  in_out_batch_strides);
                }
            };

            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                const index_t grid_size_a =
                    arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                        arg.a_in_transpose_desc_);
                const index_t grid_size_b =
                    arg.elementwise_block_2_ctile_map_transpose_b_.CalculateGridSize(
                        arg.b_in_transpose_desc_);

                ADataType* p_a_out_grid = type_convert<ADataType*>(arg.p_workspace_) +
                                          arg.GetWorkspaceETensorSizeBytes() / sizeof(ADataType);
                BDataType* p_b_out_grid =
                    type_convert<BDataType*>(arg.p_workspace_) +
                    (arg.GetWorkspaceETensorSizeBytes() + arg.GetWorkspaceATensorSizeBytes()) /
                        sizeof(BDataType);

                // Different data type for A and B is not supported
                auto kernel_transpose = kernel_elementwise_dual<GridwiseElementwiseTranspose,
                                                                GridwiseElementwiseTranspose,
                                                                ck::Tuple<NGCHWTransposeDescType>,
                                                                ck::Tuple<NGCHWTransposeDescType>,
                                                                ck::Tuple<NHWGCTransposeDescType>,
                                                                ck::Tuple<NHWGCTransposeDescType>,
                                                                ck::Tuple<const ADataType*>,
                                                                ck::Tuple<const ADataType*>,
                                                                ck::Tuple<ADataType*>,
                                                                ck::Tuple<ADataType*>,
                                                                Block2TileMapElementwise,
                                                                Block2TileMapElementwise,
                                                                element_wise::PassThrough>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_transpose,
                                                   dim3(grid_size_a + grid_size_b),
                                                   dim3(BlockSize),
                                                   0,
                                                   make_tuple(arg.a_in_transpose_desc_),
                                                   make_tuple(arg.b_in_transpose_desc_),
                                                   make_tuple(arg.a_out_transpose_desc_),
                                                   make_tuple(arg.b_out_transpose_desc_),
                                                   make_tuple(arg.p_a_grid_),
                                                   make_tuple(arg.p_b_grid_),
                                                   make_tuple(p_a_out_grid),
                                                   make_tuple(p_b_out_grid),
                                                   arg.elementwise_block_2_ctile_map_transpose_a_,
                                                   arg.elementwise_block_2_ctile_map_transpose_b_,
                                                   element_wise::PassThrough{},
                                                   grid_size_a);
            }

            avg_time += RunGemmV3<GridwiseGemm>(arg, stream_config);
            avg_time += launch_elementwise_kernel();
            return avg_time;
        }

        INVOKER_RUN_IMPL

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        const index_t GemmM = arg.a_grid_desc_k0_m_k1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_k0_n_k1_.GetLength(I1);
        const index_t GemmK =
            arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

        if constexpr(is_same_v<ComputeTypeA, ck::tf32_t> || is_same_v<ComputeTypeB, ck::tf32_t>)
        {
            if(!is_tf32_supported())
            {
                return false;
            }
            if constexpr(!is_same_v<ComputeTypeA, ComputeTypeB>)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "ComputeDataType for A and B should be same while using TF32"
                              << std::endl;
                }
                return false;
            }
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                typename GridwiseGemm64::Argument gemm_arg{
                    nullptr, nullptr, nullptr, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

                const auto num_k_loop = gemm_arg.AK0 / (KPerBlock / K1);
                if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
                {
                    if(num_k_loop <= GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages)
                    {
                        return false;
                    }
                }
            }
            else
            {
                return false;
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                typename GridwiseGemm32::Argument gemm_arg{
                    nullptr, nullptr, nullptr, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

                const auto num_k_loop = gemm_arg.AK0 / (KPerBlock / K1);
                if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
                {
                    if(num_k_loop <= GridwiseGemm32::BlockwiseGemmPipe::PrefetchStages)
                    {
                        return false;
                    }
                }
            }
            else
            {
                return false;
            }
            // TODO: this is needed because there is a bug
            if(arg.k_batch_ > 1)
            {
                return false;
            }
        }

        // Check this here, it allows to use other instances from factory even
        // if workspace is not allocated
        if(!arg.p_workspace_)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Warning: Workspace for "
                             "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument is not "
                             "allocated, use SetWorkSpacePointer."
                          << std::endl;
            }
            return false;
        }
        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        if constexpr(NumGroupsToMerge > 1)
        {
            // support only if whole M and N can be proccessed on one block
            if(!(GemmM <= MPerBlock && GemmN <= NPerBlock))
            {
                return false;
            }
            if(!(arg.Conv_C_ == 1 && arg.Conv_K_ == 1))
            {
                return false;
            }
            if(arg.Conv_G_ % NumGroupsToMerge != 0)
            {
                return false;
            }
        }

        const bool is_w_pad_zero = arg.input_left_pads_[NDimSpatial - 1] == 0 &&
                                   arg.input_right_pads_[NDimSpatial - 1] == 0;
        const auto X                 = arg.filter_spatial_lengths_[NDimSpatial - 1];
        const bool XC_access_allowed = arg.Conv_G_ == 1 &&
                                       (arg.Conv_C_ * X) % BBlockTransferSrcScalarPerVector == 0 &&
                                       is_w_pad_zero;

        if(!((arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0 || XC_access_allowed) &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0))
        {
            if(!(arg.Conv_K_ == 1 && arg.compute_ptr_offset_of_batch_.BatchStrideA_ == 1 &&
                 NumGroupsToMerge > 1))
            {
                return false;
            }
            if(!(arg.Conv_C_ == 1 && arg.compute_ptr_offset_of_batch_.BatchStrideB_ == 1 &&
                 NumGroupsToMerge > 1))
            {
                return false;
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 1 && BBlockTransferSrcVectorDim == 1))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                     is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
        {
            if((arg.Conv_G_ * arg.Conv_C_) % TransposeTransferDstScalarPerVector != 0)
            {
                return false;
            }

            if((arg.Conv_G_ * arg.Conv_K_) % TransposeTransferDstScalarPerVector != 0)
            {
                return false;
            }

            const index_t input_spatial_acum = ck::accumulate_n<index_t>(
                arg.input_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
            const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                arg.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

            if(input_spatial_acum % TransposeTransferSrcScalarPerVector != 0)
            {
                return false;
            }

            if(output_spatial_acum % TransposeTransferSrcScalarPerVector != 0)
            {
                return false;
            }

            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            if(!(arg.a_out_transpose_desc_.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
                 arg.b_out_transpose_desc_.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB))
            {
                return false;
            }
        }

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << CBlockTransferScalarPerVector_NWaveNPerXdl << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << NumGroupsToMerge;
            
        if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() || 
                        is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>()) {
                str << ", TransposeTransferSrcScalarPerVector: "
                << TransposeTransferSrcScalarPerVector <<", "
                << "TransposeTransferDstScalarPerVector: " << TransposeTransferDstScalarPerVector;
            }

            
            str << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<DeviceOp>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
        return std::make_unique<ck_tile::reflect::InstanceStringDescription>(GetInstanceString());
    }
#endif
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
