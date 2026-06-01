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
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        const index_t num_k_per_block)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    if constexpr(CGlobalMemoryDataOperation != InMemoryDataOperationEnum::AtomicAdd)
    {
#endif

        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>();
        __shared__ char p_shared[LDS_size];

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<AGridDesc_AK0_M_K1,
                                   BGridDesc_BK0_N_K1,
                                   CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                   ComputePtrOffsetOfBatch,
                                   1,
                                   HasMainKBlockLoop,
                                   CGlobalMemoryDataOperation,
                                   TailNum>(p_shared,
                                            a_grid_desc_ak0_m_ak1,
                                            b_grid_desc_bk0_n_bk1,
                                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                                            compute_ptr_offset_of_batch,
                                            num_k_per_block,
                                            karg,
                                            epilogue_args);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = compute_ptr_offset_of_batch;
    ignore = num_k_per_block;
#endif // end of if (defined(__gfx9__)
}

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
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
          ck::index_t KPerBlock,
          ck::index_t ABK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = InDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3
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
    using DeviceOp = DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3;

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using EDataType = WeiDataType;

    static constexpr index_t NumDTensor = DsLayout::Size();

    using AElementwiseOperation   = OutElementwiseOperation;
    using BElementwiseOperation   = InElementwiseOperation;
    using CDEElementwiseOperation = WeiElementwiseOperation;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;
    static constexpr auto ABK1Number             = Number<ABK1>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemmV2<NDimSpatial,
                                       MPerBlock,
                                       NPerBlock,
                                       ABK1Number,
                                       KPerBlock / ABK1Number,
                                       1 /*NumGroupsToMerge*/,
                                       ConvBackwardWeightSpecialization>{};

    static constexpr index_t MaxScalarPerVectorFP32 = 4;
    static constexpr index_t WorkspaceInOutScalarPerVector =
        is_same_v<AccDataType, float>
            ? math::min(CShuffleBlockTransferScalarPerVector_NPerBlock, MaxScalarPerVectorFP32)
            : CShuffleBlockTransferScalarPerVector_NPerBlock;

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

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        Tuple<>,
        tensor_layout::gemm::RowMajor,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        AccDataType,
        Tuple<>,
        AccDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        element_wise::PassThrough, // CDEElementwiseOperations
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        ABK1,
        ABK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<WorkspaceInOutScalarPerVector>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false, // permuteA
        false, // permuteB
        false, // IsBPreShuffled
        true>; // ForceThreadTileTransfer

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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);
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
                            Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
                            I1,
                            I1>;

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}, 1, 1));

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
                ck::index_t gemmM, gemmN, gemmK;
                std::tie(gemmM, gemmN, gemmK) =
                    get_bwd_weight_gemm_sizes<NDimSpatial>(a_g_n_k_wos_lengths, e_g_k_c_xs_lengths);

                const auto grid_size =
                    calculate_mn_grid_size<MPerBlock, NPerBlock>(gemmM, gemmN) * Conv_G_;
                k_batch_ = get_best_occupancy_k_batch_value(active_workgroups_per_cu.max_occupancy_,
                                                            grid_size);

                // Ensure that k_batch_ does not exceed the maximum value
                // for the GEMM pipeline.
                const auto k_batch_max = math::integer_divide_ceil((gemmK - 1), KPerBlock);
                k_batch_               = std::min(k_batch_, k_batch_max);

                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[SPLIT-K AUTODEDUCE] k_batch max value: " << k_batch_max
                              << std::endl;
                    std::cout << "[SPLIT-K AUTODEDUCE] Final k_batch value: " << k_batch_
                              << std::endl;
                }
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

            const index_t GemmM = a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);
            const index_t GemmN = b_grid_desc_kbatch_k0_n_k1_.GetLength(I1);

            c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                GridwiseGemm::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ce_grid_desc_m_n_,
                    GridwiseGemm::CalculateMBlock(GemmM),
                    GridwiseGemm::CalculateNBlock(GemmN));
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
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;
        DsGridDesc_M_N ds_grid_descs_tuple_;

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

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float ave_time = 0;

            const index_t GemmM = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1);
            const index_t GemmK = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) *
                                  arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2);

            AccDataType* p_e_grid = type_convert<AccDataType*>(arg.p_workspace_);

            // Convolution kernel dispatch
            typename GridwiseGemm::Argument gemm_arg{
                std::array<const void*, 1>{arg.p_a_grid_},
                std::array<const void*, 1>{arg.p_b_grid_},
                std::array<const void*, 0>{}, // p_ds_grid_
                p_e_grid,
                GemmM,
                GemmN,
                GemmK,
                std::array<index_t, 1>{I0},
                std::array<index_t, 1>{I0},
                std::array<index_t, 0>{}, // StrideDs_
                I0,
                arg.k_batch_,
                AElementwiseOperation{},
                BElementwiseOperation{},
                element_wise::PassThrough{}}; // CElementwiseOperation

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
                gemm_arg.M, gemm_arg.N, gemm_arg.KBatch, arg.Conv_G_);

            index_t k_grain                  = gemm_arg.KBatch * KPerBlock;
            index_t K_split                  = (gemm_arg.K + k_grain - 1) / k_grain * KPerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto num_k_per_block =
                arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(Number<0>{}) / gemm_arg.KBatch;

            const auto clear_workspace = [&]() {
                hip_check_error(hipMemsetAsync(
                    p_e_grid, 0, arg.GetWorkspaceSizeBytes(), stream_config.stream_id_));
            };

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;

                    std::array<std::size_t, GridwiseGemm::NumATensor> size_as_buffers;
                    size_as_buffers[0] = arg.a_grid_desc_kbatch_k0_m_k1_.GetElementSpaceSize() *
                                         sizeof(ADataType) / GridwiseGemm::APackedSize;

                    std::array<std::size_t, GridwiseGemm::NumBTensor> size_bs_buffers;
                    size_bs_buffers[0] = arg.b_grid_desc_kbatch_k0_n_k1_.GetElementSpaceSize() *
                                         sizeof(BDataType) / GridwiseGemm::BPackedSize;

                    std::array<std::size_t, 0> size_ds_buffers;

                    ck::utility::RotatingMemWrapperMultiABD<typename GridwiseGemm::Argument,
                                                            Tuple<ADataType>,
                                                            Tuple<BDataType>,
                                                            Tuple<>>
                        rotating_mem(gemm_arg_,
                                     stream_config.rotating_count,
                                     size_as_buffers,
                                     size_bs_buffers,
                                     size_ds_buffers);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                    };
                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_kbatch_k0_m_k1_,
                        arg.b_grid_desc_kbatch_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block);
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
                        arg.a_grid_desc_kbatch_k0_m_k1_,
                        arg.b_grid_desc_kbatch_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block);
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
                        const auto kernel = kernel_grouped_conv_bwd_weight_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                else
                {
                    // TODO: Implement
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

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

            ave_time += launch_elementwise_kernel();

            return ave_time;
        }

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

        const index_t GemmM = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1);
        const index_t GemmK = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) *
                              arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2);

        typename GridwiseGemm::Argument gemm_arg{std::array<const void*, 1>{nullptr}, // p_as_grid
                                                 std::array<const void*, 1>{nullptr}, // p_bs_grid
                                                 std::array<const void*, 0>{},        // p_ds_grid
                                                 nullptr,                             // p_e_grid
                                                 GemmM,                               // M
                                                 GemmN,                               // N
                                                 GemmK,                               // K
                                                 std::array<index_t, 1>{I0},          // StrideAs
                                                 std::array<index_t, 1>{I0},          // StrideBs
                                                 std::array<index_t, 0>{},            // StrideDs
                                                 I0,                                  // StrideE
                                                 arg.k_batch_,
                                                 AElementwiseOperation{},
                                                 BElementwiseOperation{},
                                                 element_wise::PassThrough{}};

        const auto num_k_loop = gemm_arg.AK0 / (KPerBlock / ABK1);
        if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
        {
            if(num_k_loop <= GridwiseGemm::BlockwiseGemmPipe::PrefetchStages)
            {
                return false;
            }
        }

        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        if(arg.k_batch_ > 1 && ck::is_gfx11_supported())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported splitK on gfx11." << std::endl;
            }
            // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
            return false;
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
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
        if(!(ABlockTransferSrcVectorDim == 1 && BBlockTransferSrcVectorDim == 1 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CShuffleBlockTransferScalarPerVector_NPerBlock == 0 &&
             arg.Conv_C_ % WorkspaceInOutScalarPerVector == 0))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(gemm_arg);
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
        str << "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << ABK1 << ", "
            << MRepeat << ", "
            << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_AK1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_BK1 << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << CShuffleBlockTransferScalarPerVector_NPerBlock
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
                "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3::Argument structure!");
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
                "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3::Argument structure!");
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(
            ck_tile::reflect::HasInstanceTraits<DeviceOp>,
            "Specialization of instance_traits not found. Please check that a "
            "specialization exists in file "
            "ck_tile/builder/reflect/"
            "instance_traits_device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp "
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
