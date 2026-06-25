// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight_multiple_d.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

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
                                          const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        const index_t num_blocks_per_batch =
            __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
        const index_t g_idx =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        __shared__ FloatA p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatA)];

        GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                      p_b_grid + b_batch_offset,
                                                      p_c_grid + c_batch_offset,
                                                      p_shared,
                                                      a_b_k0_m_k1_grid_desc,
                                                      b_b_k0_n_k1_grid_desc,
                                                      c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op,
                                                      block_2_ctile_map);
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

    compute_ptr_offset_of_batch.GetAPtrOffset(0);
    compute_ptr_offset_of_batch.GetBPtrOffset(0);
    compute_ptr_offset_of_batch.GetCPtrOffset(0);
#endif // end of if (defined(__gfx9__))
}

template <index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename DsDataType,
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
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle
    : public DeviceGroupedConvBwdWeightMultipleD<NDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 OutLayout,
                                                 DsLayout,
                                                 InDataType,
                                                 WeiDataType,
                                                 OutDataType,
                                                 DsDataType,
                                                 InElementwiseOperation,
                                                 WeiElementwiseOperation,
                                                 OutElementwiseOperation,
                                                 ComputeTypeA,
                                                 ComputeTypeB>
{
    using DeviceOp = DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using EDataType = WeiDataType;

    static constexpr index_t NumDTensor = DsLayout::Size();

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

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemm<NDimSpatial,
                                     MPerBlock,
                                     NPerBlock,
                                     K1Number,
                                     K0PerBlock,
                                     ConvBackwardWeightSpecialization>{};

    static constexpr index_t MaxScalarPerVectorFP32 = 4;
    static constexpr index_t WorkspaceInOutScalarPerVector =
        is_same_v<AccDataType, float>
            ? math::min(CBlockTransferScalarPerVector_NWaveNPerXdl, MaxScalarPerVectorFP32)
            : CBlockTransferScalarPerVector_NWaveNPerXdl;

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

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_bwd_weight<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        AccDataType,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        element_wise::PassThrough,
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
        WorkspaceInOutScalarPerVector,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true,
        true,
        1,
        PipelineVersion::v1,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    static constexpr auto MakeElementwiseInputSequence()
    {
        return generate_sequence_v2(
            [&](auto) constexpr { return Number<WorkspaceInOutScalarPerVector>{}; },
            Number<NumDTensor + 1>{});
    }

    static constexpr auto GetDsGridPointerTuple()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K       = ds_g_k_c_xs_lengths[i][I1];
                const index_t C       = ds_g_k_c_xs_lengths[i][I2];
                const index_t X       = ds_g_k_c_xs_lengths[i][I3];
                const index_t CStride = ds_g_k_c_xs_strides[I2];
                const index_t KStride = ds_g_k_c_xs_strides[I1];

                const auto wei_grid_desc = make_naive_tensor_descriptor(
                    make_tuple(K, X * C), make_tuple(KStride, CStride));

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X;
                    const auto PadGemmM =
                        GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN =
                        GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K = ds_g_k_c_xs_lengths[i][I1];
                const index_t C = ds_g_k_c_xs_lengths[i][I2];
                const index_t Y = ds_g_k_c_xs_lengths[i][I3];
                const index_t X = ds_g_k_c_xs_lengths[i][I4];

                const auto wei_grid_desc =
                    conv_to_gemm_transformer.template make_wei_grid_desc<NDim>(
                        K, Y, X, C, ds_g_k_c_xs_strides[i]);

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X * Y;
                    const auto PadGemmM =
                        GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN =
                        GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K = ds_g_k_c_xs_lengths[i][I1];
                const index_t C = ds_g_k_c_xs_lengths[i][I2];
                const index_t Z = ds_g_k_c_xs_lengths[i][I3];
                const index_t Y = ds_g_k_c_xs_lengths[i][I4];
                const index_t X = ds_g_k_c_xs_lengths[i][I5];

                const auto wei_grid_desc =
                    conv_to_gemm_transformer.template make_wei_grid_desc<NDim>(
                        K, Z, Y, X, C, ds_g_k_c_xs_strides[i]);

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X * Y * Z;
                    const auto PadGemmM =
                        GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN =
                        GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <typename ComputePtrOffsetOfBatch>
    static void
    InitElementwiseBatchStrides(const ComputePtrOffsetOfBatch& compute_ptr_offset_of_batch_,
                                std::array<index_t, NumDTensor + I1>& input_batch_strides,
                                std::array<index_t, I1>& output_batch_strides)
    {
        input_batch_strides[I0]  = compute_ptr_offset_of_batch_.BatchStrideC_;
        output_batch_strides[I0] = compute_ptr_offset_of_batch_.BatchStrideC_;

        // input_batch_strides = {C, Ds...}
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            input_batch_strides[i + 1] = compute_ptr_offset_of_batch_.BatchStrideDs_[i];
        });
    }

    using DsGridDesc_M_N     = decltype(MakeDsGridDescriptor_M_N<NDimSpatial>({}, {}));
    using CDGridDesc_M_N     = decltype(concat_tuple(Tuple<CGridDesc_M_N>{}, DsGridDesc_M_N{}));
    using DsGridPointerTuple = decltype(GetDsGridPointerTuple());
    using CDDataTypes   = decltype(concat_tuple(Tuple<const AccDataType*>{}, DsGridPointerTuple{}));
    using EGridDesc_M_N = CGridDesc_M_N;
    static constexpr index_t ClusterLengthMPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);
    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    using GridwiseElementwise =
        GridwiseElementwise<CDGridDesc_M_N,
                            Tuple<EGridDesc_M_N>,
                            CDDataTypes,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            CDEElementwiseOperation,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<0, 1>,
                            decltype(MakeElementwiseInputSequence()),
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            I1,
                            I1>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm64::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));

    using Block2CTileMap =
        decltype(GridwiseGemm64::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    struct ActiveWorkgroupsPerCU
    {
        template <typename GridwiseGemm>
        int GetMaxOccupancy()
        {
            constexpr int dynamic_smem_size = 0;
            int max_occupancy               = 0;
            hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_occupancy,
                kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    AccDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    element_wise::PassThrough,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                    true>,
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
        Argument(
            const InDataType* p_in_grid,
            WeiDataType* p_wei_grid,
            const OutDataType* p_out_grid,
            const std::array<const void*, NumDTensor>& p_ds,
            const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
            const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
            const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
            const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
            const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
            const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
            const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
            const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
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
              p_ds_grid_{},
              p_e_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              ce_grid_desc_m_n_{},
              block_2_ctile_map_{},
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

            const auto descs =
                conv_to_gemm_transformer
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
                        k_batch_);

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                static_assert(is_same_v<DLayout, WeiLayout>, "Not supported D data layout");

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_k_c_xs_strides[i][0];
            });

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            ce_grid_desc_m_n_           = descs[I2];

            ds_grid_descs_tuple_ =
                MakeDsGridDescriptor_M_N<NDimSpatial>(ds_g_k_c_xs_lengths, ds_g_k_c_xs_strides);

            block_2_ctile_map_ =
                GridwiseGemm64::MakeCBlockClusterAdaptor(ce_grid_desc_m_n_, M01, N01, k_batch_);
            elementwise_block_2_ctile_map_ = Block2TileMapElementwise{
                ce_grid_desc_m_n_.GetLength(I0), ce_grid_desc_m_n_.GetLength(I1)};

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{});
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return sizeof(AccDataType) * ce_grid_desc_m_n_.GetElementSpaceSize() * Conv_G_;
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        DsGridPointerTuple p_ds_grid_;
        EDataType* p_e_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N ce_grid_desc_m_n_;
        DsGridDesc_M_N ds_grid_descs_tuple_;

        Block2CTileMap block_2_ctile_map_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;

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

            std::cout << "arg.ce_grid_desc_m_n_{" << arg.ce_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.ce_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.ce_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
            }

            auto c_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.ce_grid_desc_m_n_);
            const auto K0                     = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);
            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            auto launch_gemm_kernel = [&](auto has_main_k_block_loop) {
                AccDataType* p_c_grid = type_convert<AccDataType*>(arg.p_workspace_);
                const index_t grid_size =
                    arg.block_2_ctile_map_.CalculateGridSize(arg.ce_grid_desc_m_n_) * arg.Conv_G_;

                constexpr bool has_main_loop = has_main_k_block_loop.value;

                auto preprocess = [&]() {
                    hip_check_error(hipMemsetAsync(
                        p_c_grid, 0, arg.c_space_size_bytes, stream_config.stream_id_));
                };

                const auto kernel = kernel_batched_gemm_xdlops_bwd_weight<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    AccDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    element_wise::PassThrough,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                    has_main_loop>;

                return launch_and_time_kernel_with_preprocess(
                    stream_config,
                    preprocess,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a_grid_,
                    arg.p_b_grid_,
                    p_c_grid,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    element_wise::PassThrough{},
                    arg.Conv_G_,
                    arg.a_grid_desc_kbatch_k0_m_k1_,
                    arg.b_grid_desc_kbatch_k0_n_k1_,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    arg.block_2_ctile_map_,
                    arg.compute_ptr_offset_of_batch_);
            };

            auto launch_elementwise_kernel = [&]() {
                const AccDataType* p_c_grid = type_convert<const AccDataType*>(arg.p_workspace_);
                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_.CalculateGridSize(arg.ce_grid_desc_m_n_) *
                    arg.Conv_G_;

                std::array<index_t, NumDTensor + I1> input_batch_strides;
                std::array<index_t, I1> output_batch_strides;
                InitElementwiseBatchStrides(
                    arg.compute_ptr_offset_of_batch_, input_batch_strides, output_batch_strides);

                const auto kernel = kernel_batched_elementwise<GridwiseElementwise,
                                                               CDGridDesc_M_N,
                                                               ck::Tuple<EGridDesc_M_N>,
                                                               CDDataTypes,
                                                               ck::Tuple<EDataType*>,
                                                               Block2TileMapElementwise,
                                                               CDEElementwiseOperation,
                                                               NumDTensor + I1,
                                                               I1>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    concat_tuple(make_tuple(arg.ce_grid_desc_m_n_), arg.ds_grid_descs_tuple_),
                    make_tuple(arg.ce_grid_desc_m_n_),
                    concat_tuple(make_tuple(p_c_grid), arg.p_ds_grid_),
                    arg.p_e_grid_,
                    arg.elementwise_block_2_ctile_map_,
                    arg.cde_element_op_,
                    arg.Conv_G_,
                    input_batch_strides,
                    output_batch_strides);
            };

            float avg_time = 0;
            if(has_main_k0_block_loop)
            {
                avg_time = launch_gemm_kernel(integral_constant<bool, true>{});
            }
            else
            {
                avg_time = launch_gemm_kernel(integral_constant<bool, false>{});
            }

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
                           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))
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
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0 &&
             arg.Conv_C_ % WorkspaceInOutScalarPerVector == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                                     arg.b_grid_desc_kbatch_k0_n_k1_,
                                                     arg.ce_grid_desc_m_n_,
                                                     arg.block_2_ctile_map_);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                                     arg.b_grid_desc_kbatch_k0_n_k1_,
                                                     arg.ce_grid_desc_m_n_,
                                                     arg.block_2_ctile_map_);
            }
        }
        return false;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const InDataType* p_in_grid,
        WeiDataType* p_wei_grid,
        const OutDataType* p_out_grid,
        const std::array<const void*, NumDTensor>& p_ds,
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
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
                        p_ds,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        ds_g_k_c_xs_lengths,
                        ds_g_k_c_xs_strides,
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

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_in_grid,
        void* p_wei_grid,
        const void* p_out_grid,
        const std::array<const void*, NumDTensor>& p_ds,
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
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
                                          p_ds,
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          ds_g_k_c_xs_lengths,
                                          ds_g_k_c_xs_strides,
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
        str << "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle"
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
            << CBlockTransferScalarPerVector_NWaveNPerXdl
            << ">";
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
                "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle::Argument structure!");
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
                "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle::Argument structure!");
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<DeviceOp>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp "
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
