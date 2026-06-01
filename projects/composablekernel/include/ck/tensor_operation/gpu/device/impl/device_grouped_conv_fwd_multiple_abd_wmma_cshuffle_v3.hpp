// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm Wmma Cshuffle V3 to realize grouped forward convolution.
 *
 * \tparam ComputePtrOffset Class that computes the base pointer offsets of A, B, D and E
 * matrices for groups or splitN. Currently it works for identical strides, but this can be extended
 * to other layouts. The returned offset can be either \p index_t or \p long_index_t. If it returns
 * \p long_index_t, we are not subject to the 2GB limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in the id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 */
template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffset, // For Batch (group) and N
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_fwd_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
        const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock,
        const ComputePtrOffset compute_ptr_offset_of_batch,
        const ComputePtrOffset compute_ptr_offset_of_n,
        const index_t num_k_per_block)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        __shared__ char p_shared[GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>()];

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<AGridDesc_AK0_M_AK1,
                                   BGridDesc_BK0_N_BK1,
                                   DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                                   EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                   ComputePtrOffset,
                                   ComputePtrOffset,
                                   HasMainKBlockLoop,
                                   EGlobalMemoryDataOperation,
                                   TailNum>(p_shared,
                                            a_grid_desc_ak0_m_ak1,
                                            b_grid_desc_bk0_n_bk1,
                                            ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                            e_grid_desc_mblock_mperblock_nblock_nperblock,
                                            compute_ptr_offset_of_batch,
                                            compute_ptr_offset_of_n,
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
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = compute_ptr_offset_of_batch;
    ignore = compute_ptr_offset_of_n;
    ignore = num_k_per_block;
#endif // End of if (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))
}

} // namespace

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Forward convolution with up to 3 spatial dimentions
//  @li         Input tensor in GNWC data format
//  @li         Weight tensor in GKXC data format
//  @li         Output tensor in GNWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType,
          index_t NumGroupsToMerge  = 1>
struct DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
    : public DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                             ALayout,
                                             BLayout,
                                             DsLayout,
                                             ELayout,
                                             ADataType,
                                             BDataType,
                                             DsDataType,
                                             EDataType,
                                             AElementwiseOperation,
                                             BElementwiseOperation,
                                             CDEElementwiseOperation,
                                             AComputeDataType,
                                             BComputeDataType>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3;

    static_assert(NumGroupsToMerge >= 1);

    static constexpr bool isMultiA  = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB  = is_detected<is_tuple, BDataType>::value;
    static constexpr bool isMultiAB = isMultiA || isMultiB;
    static constexpr bool isMultiD  = DsDataType::Size() > 0;

    // Note: I don't think this case ever occurs.
    static constexpr bool isMultiABD = isMultiA && isMultiB && isMultiD;

    // NGCHW is not supported for multiAB.
    static_assert(!(is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                    is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()) ||
                  !(isMultiA || isMultiB));

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    // TODO: This parameter is no longer supported by Gridwise!
    // static constexpr bool DoElementwiseBeforeCShuffle =
    //     !isMultiD && is_same_v<EDataType, bhalf_t> &&
    //     !is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr bool isATensorColMajor =
        (ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter1x1Stride1Pad0) &&
        (ABlockTransferSrcVectorDim == 1) && (NumGroupsToMerge == 1) &&
        (is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
         is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>());

    static constexpr bool NeedTransposeKernel =
        (isATensorColMajor == false) && (is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                                         is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>());

    static constexpr bool CTranspose = (NeedTransposeKernel == false) && (isMultiAB == false) &&
                                       (is_same_v<ELayout, tensor_layout::convolution::NGKHW> ||
                                        is_same_v<ELayout, tensor_layout::convolution::NGKDHW>);

    // Generate vector size for C & Ds
    using CDEBlockTransferScalarPerVectors =
        typename uniform_sequence_gen<NumDTensor + 1,
                                      CDEBlockTransferScalarPerVector_NPerBlock>::type;

    using ConvToGemmFwdTransformer = TransformConvFwdToGemm<NDimSpatial,
                                                            ConvForwardSpecialization,
                                                            true /*SplitN*/,
                                                            ADataType,
                                                            EDataType,
                                                            NumGroupsToMerge,
                                                            index_t,
                                                            CTranspose>;

    using ComputePtrOffset = ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto conv_ngchw_to_nhwgc_transformer =
        TransformConvNGCHWToNHWGC<ALayout,
                                  BLayout,
                                  ELayout,
                                  NDimSpatial,
                                  MPerBlock / ClusterLengthNPerBlock,
                                  NPerBlock / ClusterLengthNPerBlock>{};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_AK0_M_AK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
             is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
             ctc::NHWGC,
             std::conditional_t<is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
                                ctc::NDHWGC,
                                ALay>>;

        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<Layout>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        const auto M = in_gemmm_gemmk_desc.GetLength(I0);
        const auto K = in_gemmm_gemmk_desc.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(in_gemmm_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_BK0_N_BK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
             is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
             ctc::GKYXC,
             std::conditional_t<is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
                                ctc::GKZYXC,
                                BLay>>;

        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<Layout>();

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        const auto N = wei_gemmn_gemmk_desc.GetLength(I0);
        const auto K = wei_gemmn_gemmk_desc.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(wei_gemmn_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
             is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
             ctc::NHWGK,
             std::conditional_t<is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>() && NeedTransposeKernel,
                                ctc::NDHWGK,
                                ELay>>;

        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<Layout>();

        if constexpr(CTranspose)
        {
            constexpr auto matrix_padder_trans =
                MatrixPadder<GemmSpec, index_t, index_t, index_t>{NPerBlock, MPerBlock, KPerBlock};
            return matrix_padder_trans.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);
        }
        else
        {
            return matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);
        }
    }

    // Shape of Ds and E must be aligned. Strides can be different.
    // Pass e_g_n_k_wos_lengths for logical broadcast.
    static auto MakeDsGridDescriptor_M_N(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer);
            },
            Number<NumDTensor>{});
    }

    // Gridwise always expects tuple of datatypes.
    using GemmAsDataType = std::conditional_t<!isMultiA, Tuple<ADataType>, ADataType>;
    using GemmBsDataType = std::conditional_t<!isMultiB, Tuple<BDataType>, BDataType>;

    // Use appropriate gridwise gemm
    // Note / TODO: After the convolution has been converted to gemm, the 2D tensor descriptors will
    // in general not be RowMajor or ColumnMajor but have a more complex layout. For now we just
    // pass RCR (or CRC for CTranspose) to the gridwise gemm. This is currently only used to
    // determine the LDS block descriptors, *IF* we are not using extraM and extraN. It seems like
    // we are able to freely set these anyway without affecting results, but RCR (or CRC for
    // CTranspose) is supposedly the most accurate (and perhaps performant). The 2D layouts are also
    // used in the gridwise CheckValidity() function, where it determines some vector access checks
    // and MNPerBlock if we are not using padding. We may not actually needs these checks but keep
    // them for now.
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        tensor_layout::gemm::RowMajor,    // See Note above
        tensor_layout::gemm::ColumnMajor, // See Note above
        DsLayout,
        tensor_layout::gemm::RowMajor, // See Note above
        GemmAsDataType,
        GemmBsDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
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
        false, // AThreadTransferSrcResetCoordinateAfterRun
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        AComputeDataType,
        BComputeDataType,
        false, // PermuteA
        false, // PermuteB
        false, // IsBPreShuffled
        true>; // ForceThreadTileTransfer

    // TODO: Previously available template param DoElementwiseBeforeCShuffle!

    // In case of CTranspose we swap the following template parameters:
    // DataType, ElementWiseOp, PerBlock, K1, PerWmma, Repeat, All block transfer params.
    using GridwiseGemmSwappedParams = GridwiseGemm_wmma_cshuffle_v3<
        tensor_layout::gemm::ColumnMajor, // See Note above
        tensor_layout::gemm::RowMajor,    // See Note above

        DsLayout,
        tensor_layout::gemm::ColumnMajor, // See Note above

        GemmBsDataType,
        GemmAsDataType,

        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,

        BElementwiseOperation,
        AElementwiseOperation,

        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,

        NPerBlock,
        MPerBlock,

        KPerBlock,

        BK1,
        AK1,

        NPerWmma,
        MPerWmma,

        NRepeat,
        MRepeat,

        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun
        BBlockLdsExtraN,

        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false, // AThreadTransferSrcResetCoordinateAfterRun
        ABlockLdsExtraM,

        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,

        BComputeDataType,
        AComputeDataType, // TODO: Swapped these but will probably never get verified because the
                          // only mixed precision instances are not NCHW.

        false, // PermuteB
        false, // PermuteA
        false, // IsBPreShuffled
        true>; // ForceThreadTileTransfer

    using GridwiseGemmCTranspose =
        std::conditional_t<CTranspose, GridwiseGemmSwappedParams, GridwiseGemm>;

    // If ADataTypes or BDataTypes is tuple, user has to pass std::array with pointers.
    using APointers =
        std::conditional_t<isMultiA, std::array<const void*, NumATensor>&, const void*>;
    using BPointers =
        std::conditional_t<isMultiB, std::array<const void*, NumBTensor>&, const void*>;

    // desc for problem definition
    constexpr static ConvToGemmFwdTransformer dummy_conv_to_gemm_transformer;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemmCTranspose::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}, 1, 1))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemmCTranspose::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}, 1, 1))>;

    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, NPerBlock>;

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

    static constexpr index_t ElementwiseBlocksize = ClusterLengthNPerBlock * ClusterLengthNPerBlock;

    using GridwiseElementwiseInputTranspose =
        GridwiseElementwise<Tuple<NGCHWTransposeDescType>,
                            Tuple<NHWGCTransposeDescType>,
                            Tuple<const ADataType*>,
                            Tuple<ADataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            NPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I1,
                            I0>;

    using GridwiseElementwiseWeightTranspose =
        GridwiseElementwise<Tuple<GKCYXTransposeDescType>,
                            Tuple<GKYXCTransposeDescType>,
                            Tuple<const BDataType*>,
                            Tuple<BDataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            NPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<1>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I0,
                            I1>;

    using GridwiseElementwiseOutputTranspose =
        GridwiseElementwise<Tuple<NHWGCTransposeDescType>,
                            Tuple<NGCHWTransposeDescType>,
                            Tuple<const EDataType*>,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            NPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<1, 0>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I0,
                            I1>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(MakeAGridDescriptor_AK0_M_AK1<ALayout>(
        dummy_conv_to_gemm_transformer))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(MakeBGridDescriptor_BK0_N_BK1<BLayout>(
        dummy_conv_to_gemm_transformer))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(APointers p_as,
                 BPointers p_bs,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
            : p_as_grid_{},
              p_bs_grid_{},
              p_ds_grid_{p_ds},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{NeedTransposeKernel
                                       ? conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                                             a_g_n_c_wis_lengths, a_g_n_c_wis_strides)
                                       : a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{NeedTransposeKernel
                                      ? conv_ngchw_to_nhwgc_transformer.TransposeWeiStrides(
                                            b_g_k_c_xs_lengths, b_g_k_c_xs_strides)
                                      : b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{NeedTransposeKernel
                                       ? conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                                             e_g_n_k_wos_lengths, e_g_n_k_wos_strides)
                                       : e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              num_group_{a_g_n_c_wis_lengths_[0]},
              conv_to_gemm_transformer_{a_g_n_c_wis_lengths_,
                                        a_g_n_c_wis_strides_,
                                        b_g_k_c_xs_lengths_,
                                        b_g_k_c_xs_strides_,
                                        e_g_n_k_wos_lengths_,
                                        e_g_n_k_wos_strides_,
                                        conv_filter_strides_,
                                        conv_filter_dilations_,
                                        input_left_pads_,
                                        input_right_pads_},
              conv_N_per_block_{conv_to_gemm_transformer_.N_},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N<ELayout>(conv_to_gemm_transformer_)},
              a_grid_desc_ak0_m_ak1_{
                  MakeAGridDescriptor_AK0_M_AK1<ALayout>(conv_to_gemm_transformer_)},
              b_grid_desc_bk0_n_bk1_{
                  MakeBGridDescriptor_BK0_N_BK1<BLayout>(conv_to_gemm_transformer_)},
              compute_ptr_offset_of_groups_{},
              compute_ptr_offset_of_n_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            // A/B/E Batch/N Stride
            compute_ptr_offset_of_groups_.BatchStrideA_ =
                CTranspose ? b_g_k_c_xs_strides_[0] * NumGroupsToMerge
                           : a_g_n_c_wis_strides_[0] * NumGroupsToMerge;
            compute_ptr_offset_of_groups_.BatchStrideB_ =
                CTranspose ? a_g_n_c_wis_strides_[0] * NumGroupsToMerge
                           : b_g_k_c_xs_strides_[0] * NumGroupsToMerge;
            compute_ptr_offset_of_n_.BatchStrideA_ =
                CTranspose ? 0 : a_g_n_c_wis_strides_[1] * conv_N_per_block_;
            compute_ptr_offset_of_n_.BatchStrideB_ =
                CTranspose ? a_g_n_c_wis_strides_[1] * conv_N_per_block_ : 0;

            // Deal with the awkward APointers / BPointers types and convert to variable length
            // array of const void pointers.
            if constexpr(isMultiA)
            {
                p_as_grid_ = p_as;
            }
            else
            {
                p_as_grid_[0] = p_as;
            }
            if constexpr(isMultiB)
            {
                p_bs_grid_ = p_bs;
            }
            else
            {
                p_bs_grid_[0] = p_bs;
            }

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                // D batch stride
                compute_ptr_offset_of_groups_.BatchStrideDs_(i) =
                    ds_g_n_k_wos_strides_[i][0] * NumGroupsToMerge;
                compute_ptr_offset_of_n_.BatchStrideDs_(i) =
                    ds_g_n_k_wos_strides_[i][1] * conv_N_per_block_;

                ConvToGemmFwdTransformer conv_to_gemm_transformer_d{a_g_n_c_wis_lengths_,
                                                                    a_g_n_c_wis_strides_,
                                                                    b_g_k_c_xs_lengths_,
                                                                    b_g_k_c_xs_strides_,
                                                                    e_g_n_k_wos_lengths_,
                                                                    ds_g_n_k_wos_strides_[i],
                                                                    conv_filter_strides_,
                                                                    conv_filter_dilations_,
                                                                    input_left_pads_,
                                                                    input_right_pads_};

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer_d);
            });

            compute_ptr_offset_of_groups_.BatchStrideE_ =
                e_g_n_k_wos_strides_[0] * NumGroupsToMerge;
            compute_ptr_offset_of_n_.BatchStrideE_ = e_g_n_k_wos_strides_[1] * conv_N_per_block_;

            if constexpr(NeedTransposeKernel)
            {
                // Use not modified base strides
                a_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        a_g_n_c_wis_lengths, a_g_n_c_wis_strides);
                a_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        a_g_n_c_wis_lengths, a_g_n_c_wis_strides);

                b_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeGKCYXTransposeDesc<NDimSpatial>(
                        b_g_k_c_xs_lengths, b_g_k_c_xs_strides);
                b_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeGKYXCTransposeDesc<NDimSpatial>(
                        b_g_k_c_xs_lengths, b_g_k_c_xs_strides);

                e_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        e_g_n_k_wos_lengths, e_g_n_k_wos_strides);
                e_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        e_g_n_k_wos_lengths, e_g_n_k_wos_strides);

                elementwise_block_2_ctile_map_transpose_a_ = Block2TileMapElementwise{
                    a_in_transpose_desc_.GetLength(I0), a_in_transpose_desc_.GetLength(I1)};
                elementwise_block_2_ctile_map_transpose_b_ = Block2TileMapElementwise{
                    b_in_transpose_desc_.GetLength(I0), b_in_transpose_desc_.GetLength(I1)};
                elementwise_block_2_ctile_map_transpose_e_ = Block2TileMapElementwise{
                    e_in_transpose_desc_.GetLength(I0), e_in_transpose_desc_.GetLength(I1)};
            }

            {
                const index_t GemmM = a_grid_desc_ak0_m_ak1_.GetLength(I1);
                const index_t GemmN = b_grid_desc_bk0_n_bk1_.GetLength(I1);
                const auto MBlock   = CTranspose ? GridwiseGemmCTranspose::CalculateMBlock(GemmN)
                                                 : GridwiseGemmCTranspose::CalculateMBlock(GemmM);
                const auto NBlock   = CTranspose ? GridwiseGemmCTranspose::CalculateNBlock(GemmM)
                                                 : GridwiseGemmCTranspose::CalculateNBlock(GemmN);

                ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemmCTranspose::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n_, MBlock, NBlock);

                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemmCTranspose::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_, MBlock, NBlock);
            }
        }

        std::size_t GetWorkspaceATensorSizeBytes() const
        {
            if constexpr(NeedTransposeKernel)
            {
                const long_index_t a_acum = ck::accumulate_n<long_index_t>(
                    a_g_n_c_wis_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>());
                // Align to 128B
                return math::integer_divide_ceil(sizeof(ADataType) * a_acum, 128) * 128;
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceBTensorSizeBytes() const
        {
            if constexpr(NeedTransposeKernel &&
                         (is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                          is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>()))
            {
                const long_index_t b_acum = ck::accumulate_n<long_index_t>(
                    b_g_k_c_xs_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>());
                // Align to 128B
                return math::integer_divide_ceil(sizeof(BDataType) * b_acum, 128) * 128;
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            if constexpr(NeedTransposeKernel)
            {
                const long_index_t e_accum = ck::accumulate_n<long_index_t>(
                    e_g_n_k_wos_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>());
                return sizeof(EDataType) * e_accum;
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

        // Calculate rotating memory buffer sizes ahead of time. The convolution to gemm
        // transformer doesn't always lead to correct GetElementSpaceSize() results for the
        // Tensor descriptor, so we have to do a bunch of ad-hoc corrections. There might be
        // a better way to do this.
        auto GetRotMemAsTensorSizeBytes() const
        {
            std::array<std::size_t, NumATensor> size_as_buffers;
            ck::index_t eff_num_group = num_group_ / NumGroupsToMerge;

            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_single = remove_cvref_t<tuple_element_t<i.value, GemmAsDataType>>;
                if constexpr(is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                             is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                             is_same_v<ALayout, tensor_layout::convolution::NDHWGC>)
                {
                    size_as_buffers[i] =
                        (a_grid_desc_ak0_m_ak1_.GetElementSpaceSize() +
                         (num_group_ - NumGroupsToMerge) * (a_g_n_c_wis_strides_[0])) *
                        sizeof(ADataType_single) / GridwiseGemm::APackedSize;
                }
                else
                {
                    if(CTranspose && a_g_n_c_wis_lengths_[I1] > 1)
                    {
                        size_as_buffers[i] = (a_grid_desc_ak0_m_ak1_.GetElementSpaceSize() +
                                              (eff_num_group - 1) * (a_g_n_c_wis_strides_[0])) *
                                             sizeof(ADataType_single) / GridwiseGemm::APackedSize;
                    }
                    else
                    {
                        size_as_buffers[i] = a_grid_desc_ak0_m_ak1_.GetElementSpaceSize() *
                                             eff_num_group * sizeof(ADataType_single) /
                                             GridwiseGemm::APackedSize;
                    }
                }
            });

            return size_as_buffers;
        }

        auto GetRotMemBsTensorSizeBytes() const
        {
            std::array<std::size_t, NumBTensor> size_bs_buffers;
            ck::index_t eff_num_group = num_group_ / NumGroupsToMerge;

            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_single = remove_cvref_t<tuple_element_t<i.value, GemmBsDataType>>;
                size_bs_buffers[i] = b_grid_desc_bk0_n_bk1_.GetElementSpaceSize() * eff_num_group *
                                     sizeof(BDataType_single) / GridwiseGemm::BPackedSize;
            });

            return size_bs_buffers;
        }

        auto GetRotMemDsTensorSizeBytes() const
        {
            std::array<std::size_t, NumDTensor> size_ds_buffers;
            ck::index_t eff_num_group = num_group_ / NumGroupsToMerge;

            // TODO: Ds packed size consideration?
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                if constexpr(is_same_v<DLayout, tensor_layout::convolution::NWGK> ||
                             is_same_v<DLayout, tensor_layout::convolution::NHWGK> ||
                             is_same_v<DLayout, tensor_layout::convolution::NDHWGK>)
                {
                    size_ds_buffers[i] =
                        (ds_grid_desc_m_n_[i].GetElementSpaceSize() +
                         (num_group_ - NumGroupsToMerge) * ds_g_n_k_wos_strides_[i][0]) *
                        sizeof(DDataType);
                }
                else
                {
                    if(CTranspose && ds_g_n_k_wos_lengths_[i][I1] > 1)
                    {
                        size_ds_buffers[i] = (ds_grid_desc_m_n_[i].GetElementSpaceSize() +
                                              (eff_num_group - 1) * (ds_g_n_k_wos_strides_[i][0])) *
                                             sizeof(DDataType);
                    }
                    else
                    {
                        size_ds_buffers[i] = ds_grid_desc_m_n_[i].GetElementSpaceSize() *
                                             eff_num_group * sizeof(DDataType);
                    }
                }
            });

            return size_ds_buffers;
        }

        void Print() const
        {
            std::cout << "A[AK0, M, AK1]: " << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "B[BK0, N, BK1]: " << b_grid_desc_bk0_n_bk1_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        std::array<const void*, NumATensor> p_as_grid_;
        std::array<const void*, NumBTensor> p_bs_grid_;
        const std::array<const void*, NumDTensor> p_ds_grid_;
        EDataType* p_e_grid_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;

        // tensor descriptors for problem definiton
        index_t num_group_;

        ConvToGemmFwdTransformer conv_to_gemm_transformer_;
        index_t conv_N_per_block_;

        // tensor descriptors for block/thread-wise copy
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // for computing batch offset
        ComputePtrOffset compute_ptr_offset_of_groups_;
        ComputePtrOffset compute_ptr_offset_of_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // block-to-e-tile map
        Block2TileMapElementwise elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_b_, elementwise_block_2_ctile_map_transpose_e_;

        NGCHWTransposeDescType a_in_transpose_desc_, e_out_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, e_in_transpose_desc_;
        GKCYXTransposeDescType b_in_transpose_desc_;
        GKYXCTransposeDescType b_out_transpose_desc_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("\033[035mCTranspose %d extraM %d extraN %d\033[0m\n",
                       CTranspose,
                       ABlockLdsExtraM,
                       BBlockLdsExtraN);
            }

            float ave_time = 0;

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            const index_t num_workgroups_per_Conv_N =
                arg.a_g_n_c_wis_lengths_[I1] / arg.conv_N_per_block_;

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) =
                CTranspose
                    ? GridwiseGemmCTranspose::CalculateGridSize(GemmN, GemmM, I1 /*arg.KBatch*/)
                    : GridwiseGemmCTranspose::CalculateGridSize(GemmM, GemmN, I1 /*arg.KBatch*/);

            gdy = arg.num_group_ / NumGroupsToMerge;
            gdz = num_workgroups_per_Conv_N;

            index_t K_split = (GemmK + KPerBlock - 1) / KPerBlock * KPerBlock;
            const bool has_main_k_block_loop =
                GridwiseGemmCTranspose::CalculateHasMainKBlockLoop(K_split);
            const TailNumber tail_num = GridwiseGemmCTranspose::CalculateKBlockLoopTailNum(K_split);

            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("\033[092mnum_loop %d, has_main_k_loop %d, tail_num %d, G chunks %d, N "
                       "chunks %d\033[0m\n",
                       K_split / KPerBlock,
                       has_main_k_block_loop,
                       static_cast<int>(tail_num),
                       gdy,
                       gdz);
            }

            std::array<const void*, NumATensor> p_as_grid = arg.p_as_grid_;
            std::array<const void*, NumBTensor> p_bs_grid = arg.p_bs_grid_;
            EDataType* p_e_grid                           = arg.p_e_grid_;

            // Transpose A and B, or just A. Not compatible with multi-AB.
            if constexpr(NeedTransposeKernel)
            {
                static_assert(NumATensor == 1, "Num A Tensor should be 1\n");
                static_assert(NumBTensor == 1, "Num B Tensor should be 1\n");

                if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                             is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
                {
                    p_as_grid[0] = type_convert<const void*>(arg.p_workspace_);
                    p_bs_grid[0] = type_convert<const void*>(
                        type_convert<const BDataType*>(arg.p_workspace_) +
                        arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType));
                    p_e_grid =
                        type_convert<EDataType*>(arg.p_workspace_) +
                        (arg.GetWorkspaceATensorSizeBytes() + arg.GetWorkspaceBTensorSizeBytes()) /
                            sizeof(EDataType);
                }
                else if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                                  is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
                {
                    p_as_grid[0] = type_convert<const void*>(arg.p_workspace_);
                    p_e_grid =
                        type_convert<EDataType*>(arg.p_workspace_) +
                        (arg.GetWorkspaceATensorSizeBytes() + arg.GetWorkspaceBTensorSizeBytes()) /
                            sizeof(EDataType);
                }
            }

            const auto Run = [&](const auto& kernel) {
                if constexpr(CTranspose)
                {
                    static_assert(NumATensor == 1, "Num A Tensor should be 1\n");
                    static_assert(NumBTensor == 1, "Num B Tensor should be 1\n");

                    typename GridwiseGemmCTranspose::Argument gemm_arg{
                        p_bs_grid, // p_bs_grid
                        p_as_grid, // p_as_grid
                        arg.p_ds_grid_,
                        p_e_grid,

                        GemmN,
                        GemmM,

                        GemmK,
                        // No need to set strides, we pass descs to kernel
                        {}, // StrideBs
                        {}, // StrideAs
                        {}, // StrideDs
                        I0, // StrideE
                        I1, // kbatch
                        arg.b_element_op_,
                        arg.a_element_op_,
                        arg.cde_element_op_};
                    // TODO: No is_reduce argument, defaults to false.

                    if(stream_config.flush_cache && !NeedTransposeKernel)
                    {
                        typename GridwiseGemmCTranspose::Argument gemm_arg_ = gemm_arg;

                        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                        {
                            printf("\033[032mUsing rotating memory num group %d eff %d!\033[0m\n",
                                   arg.num_group_,
                                   arg.num_group_ / NumGroupsToMerge);
                        }

                        ck::utility::RotatingMemWrapperMultiABD<
                            typename GridwiseGemmCTranspose::Argument,
                            GemmBsDataType,
                            GemmAsDataType,
                            DsDataType>
                            rotating_mem(gemm_arg_,
                                         stream_config.rotating_count,
                                         arg.GetRotMemBsTensorSizeBytes(),
                                         arg.GetRotMemAsTensorSizeBytes(),
                                         arg.GetRotMemDsTensorSizeBytes());

                        rotating_mem.Print();

                        auto run_flush_cache = [&]() {
                            // flush icache
                            ck::utility::flush_icache();
                            // rotating mem
                            rotating_mem.Next();
                            // clear E mem

                            // TODO: The calculation of the E buffer size may not be correct in all
                            // cases, for example if the memory is not contiguous due to padding or
                            // unusual strides. Investigate when implementing splitK. It may be
                            // safer to use GetElementSpaceSize().

                            // if(arg_.KBatch > 1)
                            //     HIP_CHECK_ERROR(hipMemsetAsync(arg_.p_e_grid,
                            //                                    0,
                            //                                    arg_.M * arg_.N *
                            //                                    sizeof(EDataType),
                            //                                    stream_config.stream_id_));
                        };

                        ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                            stream_config,
                            run_flush_cache,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            gemm_arg_,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_,
                            KPerBlock); // TODO: splitK consideration (num_k_per_block)
                    }
                    else
                    {
                        ave_time += launch_and_time_kernel(
                            stream_config,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            gemm_arg,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_,
                            KPerBlock); // TODO: splitK consideration (num_k_per_block)
                    }
                }
                else
                {
                    typename GridwiseGemm::Argument gemm_arg{
                        p_as_grid, // p_as_grid
                        p_bs_grid, // p_bs_grid
                        arg.p_ds_grid_,
                        p_e_grid,
                        GemmM,
                        GemmN,
                        GemmK,
                        // No need to set strides, we pass descs to kernel
                        {}, // StrideAs
                        {}, // StrideBs
                        {}, // StrideDs
                        I0, // StrideE
                        I1, // kbatch
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_};
                    // TODO: No is_reduce argument, defaults to false.

                    if(stream_config.flush_cache && !NeedTransposeKernel)
                    {
                        typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;

                        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                        {
                            printf("\033[032mUsing rotating memory num group %d eff %d!\033[0m\n",
                                   arg.num_group_,
                                   arg.num_group_ / NumGroupsToMerge);
                        }

                        ck::utility::RotatingMemWrapperMultiABD<typename GridwiseGemm::Argument,
                                                                GemmAsDataType,
                                                                GemmBsDataType,
                                                                DsDataType>
                            rotating_mem(gemm_arg_,
                                         stream_config.rotating_count,
                                         arg.GetRotMemAsTensorSizeBytes(),
                                         arg.GetRotMemBsTensorSizeBytes(),
                                         arg.GetRotMemDsTensorSizeBytes());

                        rotating_mem.Print();

                        auto run_flush_cache = [&]() {
                            // flush icache
                            ck::utility::flush_icache();
                            // rotating mem
                            rotating_mem.Next();
                            // clear c mem

                            // TODO: The calculation of the E buffer size may not be correct in all
                            // cases, for example if the memory is not contiguous due to padding or
                            // unusual strides. Investigate when implementing splitK. It may be
                            // safer to use GetElementSpaceSize().

                            // if(arg_.KBatch > 1)
                            //     HIP_CHECK_ERROR(hipMemsetAsync(arg_.p_e_grid,
                            //                                    0,
                            //                                    arg_.M * arg_.N *
                            //                                    sizeof(EDataType),
                            //                                    stream_config.stream_id_));
                        };

                        ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                            stream_config,
                            run_flush_cache,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            gemm_arg_,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_,
                            KPerBlock); // TODO: splitK consideration (num_k_per_block)
                    }
                    else
                    {
                        ave_time += launch_and_time_kernel(
                            stream_config,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            gemm_arg,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_,
                            KPerBlock); // TODO: splitK consideration (num_k_per_block)
                    }
                }
            };

            auto CreateAndRunKernel = [&](auto has_main_k_block_loop_, auto tail_number_) {
                constexpr bool has_loop = decltype(has_main_k_block_loop_)::value;
                constexpr TailNumber tn = tail_number_;

                if constexpr(CTranspose)
                {
                    const auto kernel = kernel_grouped_conv_fwd_wmma_cshuffle_v3<
                        GridwiseGemmCTranspose,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        ComputePtrOffset,
                        has_loop, // HasMainKBlockLoop
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy,
                        tn>; // TailNumber TailNum = TailNumber::Full
                    Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_grouped_conv_fwd_wmma_cshuffle_v3<
                        GridwiseGemm,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        ComputePtrOffset,
                        has_loop, // HasMainKBlockLoop
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy,
                        tn>; // TailNumber TailNum = TailNumber::Full
                    Run(kernel);
                }
            };

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
            {
                if(has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, true>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else
                {
                    printf("Invalid has_main_k_block_loop and tail_num combination for V1!\n");
                    return 0.0f;
                }
            }
            else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
            {
                if(has_main_k_block_loop && tail_num == TailNumber::Full)
                {
                    CreateAndRunKernel(std::integral_constant<bool, true>{},
                                       std::integral_constant<TailNumber, TailNumber::Full>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Even)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Even>{});
                }
                else if(!has_main_k_block_loop && tail_num == TailNumber::Odd)
                {
                    CreateAndRunKernel(std::integral_constant<bool, false>{},
                                       std::integral_constant<TailNumber, TailNumber::Odd>{});
                }
                else
                {
                    printf("Invalid has_main_k_block_loop and tail_num combination for V3!\n");
                    return 0.0f;
                }
            }
            else
            {
                printf("Invalid pipeline version!\n");
                return 0.0f;
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;
            if constexpr(!isMultiABD)
            {
                // At least transpose A from NGCHW to NHWGC, and if necessary transpose B from GKCYX
                // to GKYXC.
                if constexpr(NeedTransposeKernel)
                {
                    static_assert(NumATensor == 1, "Num A Tensor should be 1\n");
                    static_assert(NumBTensor == 1, "Num B Tensor should be 1\n");

                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        printf("\033[32mPerforming transpose forward\033[0m\n");
                    }
                    const index_t a_grid_size =
                        arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                            arg.a_in_transpose_desc_);
                    const index_t b_grid_size =
                        (is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
                            ? arg.elementwise_block_2_ctile_map_transpose_b_.CalculateGridSize(
                                  arg.b_in_transpose_desc_)
                            : 0; // Dont run transpose B if not needed

                    ADataType* p_a_out_grid = type_convert<ADataType*>(arg.p_workspace_);
                    BDataType* p_b_out_grid =
                        type_convert<BDataType*>(arg.p_workspace_) +
                        arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);

                    auto kernel_transpose =
                        kernel_elementwise_dual<GridwiseElementwiseInputTranspose,
                                                GridwiseElementwiseWeightTranspose,
                                                ck::Tuple<NGCHWTransposeDescType>,
                                                ck::Tuple<GKCYXTransposeDescType>,
                                                ck::Tuple<NHWGCTransposeDescType>,
                                                ck::Tuple<GKYXCTransposeDescType>,
                                                ck::Tuple<const ADataType*>,
                                                ck::Tuple<const BDataType*>,
                                                ck::Tuple<ADataType*>,
                                                ck::Tuple<BDataType*>,
                                                Block2TileMapElementwise,
                                                Block2TileMapElementwise,
                                                element_wise::PassThrough>;

                    avg_time += launch_and_time_kernel(
                        stream_config,
                        kernel_transpose,
                        dim3(a_grid_size + b_grid_size),
                        dim3(ElementwiseBlocksize),
                        0,
                        make_tuple(arg.a_in_transpose_desc_),
                        make_tuple(arg.b_in_transpose_desc_),
                        make_tuple(arg.a_out_transpose_desc_),
                        make_tuple(arg.b_out_transpose_desc_),
                        make_tuple(type_convert<const ADataType*>(arg.p_as_grid_[0])),
                        make_tuple(type_convert<const BDataType*>(arg.p_bs_grid_[0])),
                        make_tuple(p_a_out_grid),
                        make_tuple(p_b_out_grid),
                        arg.elementwise_block_2_ctile_map_transpose_a_,
                        arg.elementwise_block_2_ctile_map_transpose_b_,
                        element_wise::PassThrough{},
                        a_grid_size);
                }

                avg_time += RunGemm(arg, stream_config);

                // Transpose result back to NGCHW
                if constexpr(NeedTransposeKernel)
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        printf("\033[32mPerforming transpose back\033[0m\n");
                    }
                    const index_t grid_size =
                        arg.elementwise_block_2_ctile_map_transpose_e_.CalculateGridSize(
                            arg.e_in_transpose_desc_);

                    const EDataType* p_e_in_grid =
                        type_convert<EDataType*>(arg.p_workspace_) +
                        (arg.GetWorkspaceATensorSizeBytes() + arg.GetWorkspaceBTensorSizeBytes()) /
                            sizeof(EDataType);

                    EDataType* p_e_out_grid = arg.p_e_grid_;

                    auto kernel_transpose = kernel_elementwise<GridwiseElementwiseOutputTranspose,
                                                               ck::Tuple<NHWGCTransposeDescType>,
                                                               ck::Tuple<NGCHWTransposeDescType>,
                                                               ck::Tuple<const EDataType*>,
                                                               ck::Tuple<EDataType*>,
                                                               Block2TileMapElementwise,
                                                               element_wise::PassThrough>;

                    avg_time +=
                        launch_and_time_kernel(stream_config,
                                               kernel_transpose,
                                               dim3(grid_size),
                                               dim3(ElementwiseBlocksize),
                                               0,
                                               make_tuple(arg.e_in_transpose_desc_),
                                               make_tuple(arg.e_out_transpose_desc_),
                                               make_tuple(p_e_in_grid),
                                               make_tuple(p_e_out_grid),
                                               arg.elementwise_block_2_ctile_map_transpose_e_,
                                               element_wise::PassThrough{});
                }
            }
            return avg_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        const index_t G = arg.b_g_k_c_xs_lengths_[I0];
        const index_t K = arg.b_g_k_c_xs_lengths_[I1];
        const index_t C = arg.b_g_k_c_xs_lengths_[I2];

        const index_t input_spatial_acum = ck::accumulate_n<index_t>(
            arg.a_g_n_c_wis_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());
        const index_t output_spatial_acum = ck::accumulate_n<index_t>(
            arg.e_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

        // Move this to runtime check to align Conv instances
        // with Conv Multiple D instances
        if constexpr(isMultiABD)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The MultiABD is not supported!" << " In " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1 &&
                     BlkGemmPipelineVer != BlockGemmPipelineVersion::v3)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported pipeline version!" << " In " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Current device does not support wmma instructions!" << " In "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(SpatialDim == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "The input parameters do not align with specialization "
                                     "Filter1x1Stride1Pad0!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(SpatialDim == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "The input parameters do not align with specialization "
                                     "Filter1x1Pad0!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter3x3)
        {
            if(C != 1)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "When using 3x3 ConvSpec C must be 1!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = arg.b_g_k_c_xs_lengths_[i + I3];

                if(filter_spatial_dim != I3)
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "Filter spatial dims do not match 3x3 ConvSpec!" << " In "
                                  << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                                  << std::endl;
                    }
                    return false;
                }
            }
        }

        if constexpr(NumGroupsToMerge > 1)
        {
            if(!(C == 1))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "When using mergegroups C must be 1!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
            if(G % NumGroupsToMerge != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Number of groups must be devisable by NumGroupsToMerge!" << " In "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
            if constexpr(!(is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() ||
                           is_NGCSpatial_GKSpatial_NGKSpatial<ALayout, BLayout, ELayout>() ||
                           is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                           is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported layout in combination with mergegroups!" << " In "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC> || is_same_v<ALayout, ctc::NGCW> ||
                     NeedTransposeKernel)
        {
            // TODO: This check originally said "ABlockTransferSrcVectorDim == 2", basically
            // blocking all instances with a value of 1. I've tried some though and they work just
            // fine. So I changed it to allow a value of 1 for now but there might be cases where
            // this does not work.
            // Check access per C
            if(!(ABlockTransferSrcVectorDim <= 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                // If not possible, check access per G
                if(!(ABlockTransferSrcVectorDim == 1 && (C == 1 || NumGroupsToMerge == 1) &&
                     (is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() ||
                      is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                      is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()) &&
                     G % ABlockTransferSrcScalarPerVector == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "[A Layout] The number of input channels is not a multiple of "
                                     "ABlockTransferSrcScalarPerVector!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else if constexpr(is_same_v<ALayout, ctc::NGCHW> || is_same_v<ALayout, ctc::NGCDHW>)
        {
            static_assert(NeedTransposeKernel == false);
            static_assert(NumGroupsToMerge == 1);

            if constexpr(ABlockTransferSrcScalarPerVector != 1)
            {
                if(ABlockTransferSrcVectorDim != 1)
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "ABlockTransferSrcVectorDim must be 1!" << " In " << __FILE__
                                  << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
                if(input_spatial_acum % ABlockTransferSrcScalarPerVector != 0)
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "[A Layout] The number of input channels is not a multiple of "
                                     "ABlockTransferSrcScalarPerVector!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported A Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::G_K_X_C> || is_same_v<BLayout, ctc::G_K_YX_C> ||
                     is_same_v<BLayout, ctc::G_K_ZYX_C> || is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC> ||
                     is_same_v<BLayout, ctc::KXGC> || is_same_v<BLayout, ctc::KYXGC> ||
                     is_same_v<BLayout, ctc::KZYXGC> || is_same_v<BLayout, ctc::GKCX> ||
                     is_same_v<BLayout, ctc::GKCYX> || is_same_v<BLayout, ctc::GKCZYX>)

        {
            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[B Layout] The number of input channels is not a multiple of "
                                 "BBlockTransferSrcScalarPerVector!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported B Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // Check vector access of Ds
        bool valid = 1;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            // FIXME: layout
            if constexpr(is_same_v<DLayout, ctc::G_NW_K> || is_same_v<DLayout, ctc::G_NHW_K> ||
                         is_same_v<DLayout, ctc::G_NDHW_K> || is_same_v<DLayout, ctc::GNWK> ||
                         is_same_v<DLayout, ctc::GNHWK> || is_same_v<DLayout, ctc::GNDHWK> ||
                         is_same_v<DLayout, ctc::NWGK> || is_same_v<DLayout, ctc::NHWGK> ||
                         is_same_v<DLayout, ctc::NDHWGK> || is_same_v<DLayout, ctc::G_K>)
            {
                if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "[D Layout] D tensor number " << i
                                  << " has a K size which is not a multiple of "
                                     "CDEBlockTransferScalarPerVector_NPerBlock!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    valid = 0;
                }

                if constexpr(is_same_v<DLayout, ctc::G_K>)
                {
                    // G and K must be the same
                    if(arg.ds_g_n_k_wos_lengths_[i][0] != arg.e_g_n_k_wos_lengths_[0] ||
                       arg.ds_g_n_k_wos_lengths_[i][2] != arg.e_g_n_k_wos_lengths_[2])
                    {
                        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                        {
                            std::cout << "[D Layout] D tensor number " << i
                                      << " shape does not match E shape! (GK case)" << " In "
                                      << __FILE__ << ":" << __LINE__
                                      << ", in function: " << __func__ << std::endl;
                        }
                        valid = 0;
                    }
                }
                else
                {
                    // E and D must have the same shape
                    for(index_t d = 0; d < NDimSpatial + 3; d++)
                    {
                        if(arg.ds_g_n_k_wos_lengths_[i][d] != arg.e_g_n_k_wos_lengths_[d])
                        {
                            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                            {
                                std::cout << "[D Layout] D tensor number " << i
                                          << " shape does not match E shape! (generic case)"
                                          << " In " << __FILE__ << ":" << __LINE__
                                          << ", in function: " << __func__ << std::endl;
                            }
                            valid = 0;
                        }
                    }
                }
            }
            else
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[D Layout] D tensor number " << i << " has an unknown layout!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                valid = 0;
            }
        });

        if(!valid)
            return false;

        if constexpr(NeedTransposeKernel)
        {
            if((G * C) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[NGCHW Layout] The G * C is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            if((G * K) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[NGCHW Layout] The G * K is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            if(input_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[NGCHW Layout] The input_spatial_acum is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            if(output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[NGCHW Layout] The output_spatial_acum is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }

            if(!arg.p_workspace_)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout
                        << "Warning: Workspace for "
                           "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3::Argument is not "
                           "allocated, use SetWorkSpacePointer."
                        << std::endl;
                }
                return false;
            }

            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            if(!(arg.a_out_transpose_desc_.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
                 arg.e_in_transpose_desc_.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[NGCHW Layout] One of the transposed vectors is exceeding 2GB "
                                 "memory size!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK> || is_same_v<ELayout, ctc::NGKW> ||
                     is_same_v<ELayout, ctc::NGKHW> || is_same_v<ELayout, ctc::NGKDHW>)
        {
            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[E Layout] The K is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported E Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // Gridwise gemm v3 doesn't verify descriptors size
        if(!arg.conv_to_gemm_transformer_.AreDescriptorsSmallerThan2GB())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout
                    << "[conv_to_gemm_transformer_] One of the descriptors is bigger than 2GB!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                    << std::endl;
            }
            return false;
        }

        // check Gridwise GEMM
        const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
        const index_t GemmK =
            arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

        if constexpr(CTranspose)
        {
            typename GridwiseGemmCTranspose::Argument gemm_arg{{nullptr},
                                                               {nullptr},
                                                               {},
                                                               nullptr,
                                                               GemmN,
                                                               GemmM,
                                                               GemmK,
                                                               {I0},
                                                               {I0},
                                                               {},
                                                               I0,
                                                               I1 /*KBatch*/,
                                                               arg.b_element_op_,
                                                               arg.a_element_op_,
                                                               arg.cde_element_op_};
            // TODO: No is_reduce argument, defaults to false.

            return GridwiseGemmCTranspose::CheckValidity(gemm_arg, true); // allow_short_v3_pipe
        }
        else
        {
            typename GridwiseGemmCTranspose::Argument gemm_arg{{nullptr},
                                                               {nullptr},
                                                               {},
                                                               nullptr,
                                                               GemmM,
                                                               GemmN,
                                                               GemmK,
                                                               {I0},
                                                               {I0},
                                                               {},
                                                               I0,
                                                               I1 /*KBatch*/,
                                                               arg.a_element_op_,
                                                               arg.b_element_op_,
                                                               arg.cde_element_op_};
            // TODO: No is_reduce argument, defaults to false.

            return GridwiseGemmCTranspose::CheckValidity(gemm_arg, true); // allow_short_v3_pipe
        }
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        APointers p_as,
        BPointers p_bs,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto
    MakeArgument(APointers p_as,
                 BPointers p_bs,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<long_index_t, NDimSpatial>& input_left_pads,
                 const std::array<long_index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths_i32,
                        a_g_n_c_wis_strides_i32,
                        b_g_k_c_xs_lengths_i32,
                        b_g_k_c_xs_strides_i32,
                        ds_g_n_k_wos_lengths_i32,
                        ds_g_n_k_wos_strides_i32,
                        e_g_n_k_wos_lengths_i32,
                        e_g_n_k_wos_strides_i32,
                        conv_filter_strides_i32,
                        conv_filter_dilations_i32,
                        input_left_pads_i32,
                        input_right_pads_i32,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        APointers p_as,
        BPointers p_bs,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) override
    {
        return std::make_unique<Argument>(p_as,
                                          p_bs,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(APointers p_as,
                        BPointers p_bs,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_lengths,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<long_index_t, NDimSpatial>& input_left_pads,
                        const std::array<long_index_t, NDimSpatial>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CDEElementwiseOperation& cde_element_op) override
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i32;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i32;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i32;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_strides_i32;
        std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
        std::array<index_t, NDimSpatial> input_left_pads_i32;
        std::array<index_t, NDimSpatial> input_right_pads_i32;

        array_convert(a_g_n_c_wis_lengths_i32, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i32, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i32[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i32[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i32, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i32, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i32, conv_filter_strides);
        array_convert(conv_filter_dilations_i32, conv_filter_dilations);
        array_convert(input_left_pads_i32, input_left_pads);
        array_convert(input_right_pads_i32, input_right_pads);

        return std::make_unique<Argument>(p_as,
                                          p_bs,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths_i32,
                                          a_g_n_c_wis_strides_i32,
                                          b_g_k_c_xs_lengths_i32,
                                          b_g_k_c_xs_strides_i32,
                                          ds_g_n_k_wos_lengths_i32,
                                          ds_g_n_k_wos_strides_i32,
                                          e_g_n_k_wos_lengths_i32,
                                          e_g_n_k_wos_strides_i32,
                                          conv_filter_strides_i32,
                                          conv_filter_dilations_i32,
                                          input_left_pads_i32,
                                          input_right_pads_i32,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
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
        str << "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << MRepeat << ", "
            << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CDEBlockTransferScalarPerVector_NPerBlock << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << NumGroupsToMerge
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
                "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle::Argument structure!");
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
                "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
