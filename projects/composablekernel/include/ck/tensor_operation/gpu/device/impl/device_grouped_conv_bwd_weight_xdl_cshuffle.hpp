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
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_offset_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

// Dispatch helper function for split-K hack - handles 2-way dispatch based on runtime flag
template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_B_K0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__device__ void DispatchBatchedGemmSplitKHack(const FloatA* p_a_grid,
                                              const FloatB* p_b_grid,
                                              FloatC* p_c_grid,
                                              void* p_shared,
                                              const AGridDesc_B_K0_M_K1& a_b_k0_m_k1_grid_desc,
                                              const BGridDesc_B_K0_N_K1& b_b_k0_n_k1_grid_desc,
                                              const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                              const AElementwiseOperation& a_element_op,
                                              const BElementwiseOperation& b_element_op,
                                              const CElementwiseOperation& c_element_op,
                                              const Block2CTileMap& block_2_ctile_map,
                                              const long_index_t split_k_stride_a,
                                              const long_index_t split_k_stride_b,
                                              bool split_k_offset_hack,
                                              index_t k_batch)
{
    if(split_k_offset_hack)
    {
        GridwiseGemm::template Run<HasMainKBlockLoop, true>(
            p_a_grid,
            p_b_grid,
            p_c_grid,
            p_shared,
            a_b_k0_m_k1_grid_desc,
            b_b_k0_n_k1_grid_desc,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map,
            split_k_stride_a,
            split_k_stride_b,
            k_batch);
    }
    else
    {
        GridwiseGemm::template Run<HasMainKBlockLoop, false>(
            p_a_grid,
            p_b_grid,
            p_c_grid,
            p_shared,
            a_b_k0_m_k1_grid_desc,
            b_b_k0_n_k1_grid_desc,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map,
            split_k_stride_a,
            split_k_stride_b,
            k_batch);
    }
}

template <typename GridwiseGemm,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename AGridDesc_B_K0_M_K1,
          typename BGridDesc_B_K0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_xdlops_bwd_weight(const FloatA* __restrict__ p_a_grid,
                                          const FloatB* __restrict__ p_b_grid,
                                          FloatC* __restrict__ p_c_grid,
                                          const AElementwiseOperation a_element_op,
                                          const BElementwiseOperation b_element_op,
                                          const CElementwiseOperation c_element_op,
                                          const index_t batch_count,
                                          const AGridDesc_B_K0_M_K1 a_b_k0_m_k1_grid_desc,
                                          const BGridDesc_B_K0_N_K1 b_b_k0_n_k1_grid_desc,
                                          const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                              c_grid_desc_mblock_mperblock_nblock_nperblock,
                                          const Block2CTileMap block_2_ctile_map,
                                          const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                          const long_index_t split_k_stride_a,
                                          const long_index_t split_k_stride_b,
                                          bool split_k_offset_hack,
                                          index_t k_batch)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        const index_t num_blocks_per_batch =
            __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
        const index_t g_idx =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

        const long_index_t a_batch_offset = compute_ptr_offset_of_batch.GetAPtrOffset(g_idx);
        const long_index_t b_batch_offset = compute_ptr_offset_of_batch.GetBPtrOffset(g_idx);
        const long_index_t c_batch_offset = compute_ptr_offset_of_batch.GetCPtrOffset(g_idx);

        __shared__ FloatA p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatA)];

        DispatchBatchedGemmSplitKHack<GridwiseGemm,
                                      FloatA,
                                      FloatB,
                                      FloatC,
                                      AGridDesc_B_K0_M_K1,
                                      BGridDesc_B_K0_N_K1,
                                      CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                      AElementwiseOperation,
                                      BElementwiseOperation,
                                      CElementwiseOperation,
                                      Block2CTileMap,
                                      HasMainKBlockLoop>(
            p_a_grid + a_batch_offset,
            p_b_grid + b_batch_offset,
            p_c_grid + c_batch_offset,
            p_shared,
            a_b_k0_m_k1_grid_desc,
            b_b_k0_n_k1_grid_desc,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map,
            split_k_stride_a,
            split_k_stride_b,
            split_k_offset_hack,
            k_batch);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_b_k0_m_k1_grid_desc;
    ignore = b_b_k0_n_k1_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = batch_count;
    ignore = block_2_ctile_map;
    ignore = compute_ptr_offset_of_batch;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
    ignore = split_k_offset_hack;
    ignore = k_batch;

    compute_ptr_offset_of_batch.GetAPtrOffset(0);
    compute_ptr_offset_of_batch.GetBPtrOffset(0);
    compute_ptr_offset_of_batch.GetCPtrOffset(0);
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
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
          ck::index_t K0PerBlock,
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
          typename ComputeTypeA                          = InDataType,
          typename ComputeTypeB                          = ComputeTypeA,
          index_t MaxTransposeTransferSrcScalarPerVector = 1,
          index_t MaxTransposeTransferDstScalarPerVector = 1>
struct DeviceGroupedConvBwdWeight_Xdl_CShuffle
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
    using DeviceOp = DeviceGroupedConvBwdWeight_Xdl_CShuffle;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

    // If NGCHW then ADataType must be equal to BDataType
    static_assert(!(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                    is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>()) ||
                  is_same_v<ADataType, BDataType>);

    using AElementwiseOperation = OutElementwiseOperation;
    using BElementwiseOperation = InElementwiseOperation;
    using CElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemm<NDimSpatial,
                                     MPerBlock,
                                     NPerBlock,
                                     K1Number,
                                     K0PerBlock,
                                     ConvBackwardWeightSpecialization>{};

    // Bytes per 32 lds bank: 32 * 4 bytes
    static constexpr auto BankLength = 128;
    static constexpr auto ElePerBank = BankLength / sizeof(ADataType);

    // M1 & M0
    static constexpr auto ABlockLdsM1PerBlock = ElePerBank / K1;
    static constexpr auto ABlockLdsM0PerBlock = MPerBlock / ABlockLdsM1PerBlock;
    static constexpr auto ABlockLdsM1Padding  = 4;

    // N1 & N0
    static constexpr auto BBlockLdsN1PerBlock = ElePerBank / K1;
    static constexpr auto BBlockLdsN0PerBlock = NPerBlock / BBlockLdsN1PerBlock;
    static constexpr auto BBlockLdsN1Padding  = 4;

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            dim,
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
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            dim,
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
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(
            dim,
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

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

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

    using Block2TileMapTranspose = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    static constexpr index_t TransposeTransferSrcScalarPerVectorAligned =
        std::min(NPerBlock / ClusterLengthNPerBlock, MaxTransposeTransferSrcScalarPerVector);
    static constexpr index_t TransposeTransferDstScalarPerVectorAligned =
        std::min(MPerBlock / ClusterLengthMPerBlock, MaxTransposeTransferDstScalarPerVector);

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

    using GridwiseInOutTranspose =
        GridwiseElementwise<Tuple<NGCHWTransposeDescType>,
                            Tuple<NHWGCTransposeDescType>,
                            Tuple<const ADataType*>,
                            Tuple<ADataType*>,
                            Block2TileMapTranspose,
                            element_wise::PassThrough,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<TransposeTransferSrcScalarPerVectorAligned>,
                            Sequence<TransposeTransferDstScalarPerVectorAligned>,
                            I1,
                            I0>;

    // NPerBlock is used for the first dim which is store dimension
    // (with CBlockTransferScalarPerVector_NWaveNPerXdl scalar per vector).
    using GridwiseElementwiseWeightTranspose =
        GridwiseElementwise<Tuple<GKYXCTransposeDescType>,
                            Tuple<GKCYXTransposeDescType>,
                            Tuple<const CDataType*>,
                            Tuple<CDataType*>,
                            Block2TileMapTranspose,
                            element_wise::PassThrough,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            Sequence<1>,
                            I1,
                            I0>;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        ABlockLdsM1PerBlock,
        ABlockLdsM0PerBlock,
        ABlockLdsM1Padding,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        BBlockLdsN1PerBlock,
        BBlockLdsN0PerBlock,
        BBlockLdsN1Padding,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true,
        true,
        1,
        PipelineVersion::v1,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm64::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));

    using Block2CTileMap =
        decltype(GridwiseGemm64::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    struct ActiveWorkgroupsPerCU
    {
        template <typename GridwiseGemm>
        static int GetMaxOccupancy()
        {
            constexpr int dynamic_smem_size = 0;
            int max_occupancy               = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_occupancy,
                kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    CDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<>,
                    false>, // HasMainKBlockLoop - both true/false give the same occupancy
                BlockSize,
                dynamic_smem_size));
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
              p_c_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              c_grid_desc_m_n_{},
              block_2_ctile_map_{},
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
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
                sizeof(WeiDataType);

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

#if !DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
            if(split_k < 0)
            {
                ck::index_t gemmM, gemmN;
                std::tie(gemmM, gemmN, std::ignore) =
                    get_bwd_weight_gemm_sizes<NDimSpatial>(a_g_n_k_wos_lengths, e_g_k_c_xs_lengths);

                const auto grid_size =
                    calculate_mn_grid_size<MPerBlock, NPerBlock>(gemmM, gemmN) * Conv_G_;
                k_batch_ = get_best_occupancy_k_batch_value(active_workgroups_per_cu.max_occupancy_,
                                                            grid_size);
            }
            else
#endif
            {
                k_batch_ = split_k;
            }

            // Create descriptors first (with hack flags temporarily set to false)
            // so we can check if element space sizes are divisible by k_batch
            const auto descs_initial =
                conv_to_gemm_transformer
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
                        false); // split_k_offset_b_hack (temporary)

            split_k_offset_hack_ =
                SplitKHackEligibility<NDimSpatial, InLayout, WeiLayout, OutLayout>::Check(
                    descs_initial[I0],
                    descs_initial[I1],
                    k_batch_,
                    Conv_N_,
                    output_spatial_lengths_,
                    K0PerBlock * K1);

            // Now create descriptors with the correct hack flag
            const auto descs =
                conv_to_gemm_transformer
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
                        split_k_offset_hack_);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            // Calculate stride using CalculateOffset method for accurate stride
            // This works correctly for any descriptor transform pipeline
            split_k_stride_a_ = a_grid_desc_kbatch_k0_m_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_a_ /= k_batch_;

            split_k_stride_b_ = b_grid_desc_kbatch_k0_n_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_b_ /= k_batch_;

            block_2_ctile_map_ =
                GridwiseGemm64::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides_transposed[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides_transposed[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ = e_g_k_c_xs_strides_transposed[0];

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

                elementwise_block_2_ctile_map_transpose_a_ = Block2TileMapTranspose{
                    a_in_transpose_desc_.GetLength(I0), a_in_transpose_desc_.GetLength(I1)};

                elementwise_block_2_ctile_map_transpose_b_ = Block2TileMapTranspose{
                    b_in_transpose_desc_.GetLength(I0), b_in_transpose_desc_.GetLength(I1)};

                elementwise_block_2_ctile_map_transpose_e_ = Block2TileMapTranspose{
                    e_in_transpose_desc_.GetLength(I0), e_in_transpose_desc_.GetLength(I1)};
            }
        }

        std::size_t GetWorkspaceATensorSizeBytes() const
        {
            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                // Align to 128B
                return math::integer_divide_ceil(
                           sizeof(ADataType) * a_in_transpose_desc_.GetElementSpaceSize(), 128) *
                       128;
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceBTensorSizeBytes() const
        {
            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                // Align to 128B
                return math::integer_divide_ceil(
                           sizeof(BDataType) * b_in_transpose_desc_.GetElementSpaceSize(), 128) *
                       128;
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            if constexpr(is_NGCHW_GKCYX_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                return sizeof(CDataType) * e_in_transpose_desc_.GetElementSpaceSize();
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return GetWorkspaceATensorSizeBytes() + GetWorkspaceBTensorSizeBytes() +
                   GetWorkspaceETensorSizeBytes();
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        Block2CTileMap block_2_ctile_map_;

        Block2TileMapTranspose elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_b_, elementwise_block_2_ctile_map_transpose_e_;

        NGCHWTransposeDescType a_in_transpose_desc_, b_in_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, b_out_transpose_desc_;

        GKYXCTransposeDescType e_in_transpose_desc_;
        GKCYXTransposeDescType e_out_transpose_desc_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<> compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;

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
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;
            CDataType* p_e_grid       = arg.p_c_grid_;

            auto c_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm64::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.c_grid_desc_m_n_);

            if constexpr(is_NGCHW_GKCYX_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                p_e_grid =
                    type_convert<CDataType*>(arg.p_workspace_) +
                    (arg.GetWorkspaceATensorSizeBytes() + arg.GetWorkspaceBTensorSizeBytes()) /
                        sizeof(CDataType);
            }

            if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                const index_t grid_size_a =
                    arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                        arg.a_in_transpose_desc_);
                const index_t grid_size_b =
                    arg.elementwise_block_2_ctile_map_transpose_b_.CalculateGridSize(
                        arg.b_in_transpose_desc_);

                p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                p_b_grid = type_convert<const BDataType*>(arg.p_workspace_) +
                           arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);
                ADataType* p_out_a_grid = type_convert<ADataType*>(arg.p_workspace_);
                BDataType* p_out_b_grid = type_convert<BDataType*>(arg.p_workspace_) +
                                          arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);

                // Different data type for A and B is not supported
                auto kernel_transpose = kernel_elementwise_dual<GridwiseInOutTranspose,
                                                                GridwiseInOutTranspose,
                                                                ck::Tuple<NGCHWTransposeDescType>,
                                                                ck::Tuple<NGCHWTransposeDescType>,
                                                                ck::Tuple<NHWGCTransposeDescType>,
                                                                ck::Tuple<NHWGCTransposeDescType>,
                                                                ck::Tuple<const ADataType*>,
                                                                ck::Tuple<const ADataType*>,
                                                                ck::Tuple<ADataType*>,
                                                                ck::Tuple<ADataType*>,
                                                                Block2TileMapTranspose,
                                                                Block2TileMapTranspose,
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
                                                   make_tuple(p_out_a_grid),
                                                   make_tuple(p_out_b_grid),
                                                   arg.elementwise_block_2_ctile_map_transpose_a_,
                                                   arg.elementwise_block_2_ctile_map_transpose_b_,
                                                   element_wise::PassThrough{},
                                                   grid_size_a);
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Conv_G_;

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    CDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<>,
                    has_main_loop>;

                const auto clear_workspace = [&]() {
                    hip_check_error(hipMemsetAsync(
                        p_e_grid, 0, arg.c_space_size_bytes, stream_config.stream_id_));
                };

                avg_time += launch_and_time_kernel_with_preprocess(
                    stream_config,
                    clear_workspace,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    p_a_grid,
                    p_b_grid,
                    p_e_grid,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.c_element_op_,
                    arg.Conv_G_,
                    arg.a_grid_desc_kbatch_k0_m_k1_,
                    arg.b_grid_desc_kbatch_k0_n_k1_,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    arg.block_2_ctile_map_,
                    arg.compute_ptr_offset_of_batch_,
                    arg.split_k_stride_a_,
                    arg.split_k_stride_b_,
                    arg.split_k_offset_hack_,
                    arg.k_batch_);
            };

            if(has_main_k0_block_loop)
            {
                launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                launch_kernel(integral_constant<bool, false>{});
            }

            if constexpr(is_NGCHW_GKCYX_NGKHW<InLayout, WeiLayout, OutLayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<InLayout, WeiLayout, OutLayout>())
            {
                const index_t grid_size_e =
                    arg.elementwise_block_2_ctile_map_transpose_e_.CalculateGridSize(
                        arg.e_in_transpose_desc_);

                const CDataType* p_e_in_grid = static_cast<const CDataType*>(p_e_grid);

                // Different data type for A and B is not supported
                auto kernel_transpose = kernel_elementwise<GridwiseElementwiseWeightTranspose,
                                                           ck::Tuple<GKYXCTransposeDescType>,
                                                           ck::Tuple<GKCYXTransposeDescType>,
                                                           ck::Tuple<const CDataType*>,
                                                           ck::Tuple<CDataType*>,
                                                           Block2TileMapTranspose,
                                                           element_wise::PassThrough>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_transpose,
                                                   dim3(grid_size_e),
                                                   dim3(BlockSize),
                                                   0,
                                                   make_tuple(arg.e_in_transpose_desc_),
                                                   make_tuple(arg.e_out_transpose_desc_),
                                                   make_tuple(p_e_in_grid),
                                                   make_tuple(arg.p_c_grid_),
                                                   arg.elementwise_block_2_ctile_map_transpose_e_,
                                                   element_wise::PassThrough{});
            }

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
#if DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
        if(arg.k_batch_ < 0)
        {
            return false;
        }
#endif
        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t>)
        {
            return false;
        }
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
        if constexpr(NDimSpatial == 1)
        {
            if constexpr(!is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>())
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>() ||
                           is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>() ||
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

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
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
            if((arg.Conv_G_ * arg.Conv_C_) % TransposeTransferDstScalarPerVectorAligned != 0)
            {
                return false;
            }

            if((arg.Conv_G_ * arg.Conv_K_) % TransposeTransferDstScalarPerVectorAligned != 0)
            {
                return false;
            }

            const index_t input_spatial_acum = ck::accumulate_n<index_t>(
                arg.input_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
            const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                arg.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

            if(input_spatial_acum % TransposeTransferSrcScalarPerVectorAligned != 0)
            {
                return false;
            }

            if(output_spatial_acum % TransposeTransferSrcScalarPerVectorAligned != 0)
            {
                return false;
            }

            if(!arg.p_workspace_)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Warning: Workspace for "
                                 "DeviceGroupedConvBwdWeight_Xdl_CShuffle::Argument is not "
                                 "allocated, use SetWorkSpacePointer."
                              << std::endl;
                }
                return false;
            }

            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            if(!(arg.a_out_transpose_desc_.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
                 arg.b_out_transpose_desc_.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB))
            {
                return false;
            }
        }

        // Gridwise GEMM size
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                                     arg.b_grid_desc_kbatch_k0_n_k1_,
                                                     arg.c_grid_desc_m_n_,
                                                     arg.block_2_ctile_map_);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                                     arg.b_grid_desc_kbatch_k0_n_k1_,
                                                     arg.c_grid_desc_m_n_,
                                                     arg.block_2_ctile_map_);
            }
        }
        return false;
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

        // clang-format off
        str << "DeviceGroupedConvBwdWeight_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
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
            << CBlockTransferScalarPerVector_NWaveNPerXdl;

        if constexpr(is_NGCHW_NGKHW<InLayout, WeiLayout, OutLayout>() || 
                        is_NGCDHW_NGKDHW<InLayout, WeiLayout, OutLayout>()) {
                str << ", TransposeTransferSrcScalarPerVectorAligned: "
                << TransposeTransferSrcScalarPerVectorAligned <<", "
                << "TransposeTransferDstScalarPerVectorAligned: " << TransposeTransferDstScalarPerVectorAligned;
            }

            
            str << ">";
        // clang-format on

        return str.str();
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<DeviceOp>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
        return std::make_unique<ck_tile::reflect::InstanceStringDescription>(GetInstanceString());
    }
#endif

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
                "DeviceGroupedConvBwdWeight_Xdl_CShuffle::Argument structure!");
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
                "DeviceGroupedConvBwdWeight_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
