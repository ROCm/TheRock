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
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/conv_describe.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for
 * \link DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the
 * computing of pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename AsPointer, // tuples if multi AB, pointers if no
          typename BsPointer,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          typename ComputePtrOffsetOfG,
          typename ComputePtrOffsetOfN,
          bool HasMainKBlockLoop,
          bool isMultiA,
          bool isMultiB,
          bool CTranspose>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle(
        AsPointer p_as_grid,
        BsPointer p_bs_grid,
        DsPointer p_ds_grid,
        EDataType* __restrict__ p_e_grid,
        AElementwiseOperation a_element_op,
        BElementwiseOperation b_element_op,
        CDEElementwiseOperation cde_element_op,
        const AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
        const BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
        const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_,
        const Block2ETileMap block_2_ctile_map,
        const ComputePtrOffsetOfG compute_ptr_offset_of_groups,
        const ComputePtrOffsetOfN compute_ptr_offset_of_n)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        // offset base pointer for each work-group
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const long_index_t e_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));
        const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);
        const auto& ds_n_offset     = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);

        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        DsPointer p_ds_grid_grp;

        static constexpr index_t NumDTensor =
            DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_n_offset[i] + ds_group_offset[i]; });

        if constexpr(isMultiA || isMultiB)
        {
            AsPointer p_as_grid_grp;
            BsPointer p_bs_grid_grp;

            const auto& as_group_offset = compute_ptr_offset_of_groups.GetAsPtrOffset(g_idx);

            // compute_ptr_offset_of_n_ not need BatchStrideB so
            // in case of MultiA is false but isMultiB is true
            // BatchStrideA_ is not tuple.
            if constexpr(isMultiA)
            {
                const auto& as_n_offset = compute_ptr_offset_of_n.GetAsPtrOffset(n_idx);

                static constexpr index_t NumATensor = AGridDesc_AK0_M_AK1::Size();
                static_for<0, NumATensor, 1>{}([&](auto i) {
                    p_as_grid_grp(i) = p_as_grid[i] + as_group_offset[i] + as_n_offset[i];
                });
            }
            else
            {
                const long_index_t a_n_offset = compute_ptr_offset_of_n.GetAPtrOffset(n_idx);
                static_for<0, 1, 1>{}([&](auto i) {
                    p_as_grid_grp(i) = p_as_grid[i] + as_group_offset[i] + a_n_offset;
                });
            }

            const auto& bs_group_offset = compute_ptr_offset_of_groups.GetBsPtrOffset(g_idx);

            static constexpr index_t NumBTensor = BGridDesc_BK0_N_BK1::Size();
            static_for<0, NumBTensor, 1>{}(
                [&](auto i) { p_bs_grid_grp(i) = p_bs_grid[i] + bs_group_offset[i]; });

            GridwiseGemm::template Run<HasMainKBlockLoop>(
                p_as_grid_grp,
                p_bs_grid_grp,
                p_ds_grid_grp,
                p_e_grid + e_group_offset + e_n_offset,
                p_shared,
                a_element_op,
                b_element_op,
                cde_element_op,
                a_grid_desc_k0_m_k1,
                b_grid_desc_k0_n_k1,
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
                block_2_ctile_map);
        }
        else
        {
            const long_index_t b_group_offset =
                CTranspose
                    ? amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx))
                    : amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
            const long_index_t a_group_offset =
                CTranspose
                    ? amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx))
                    : amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
            const long_index_t b_n_offset =
                CTranspose ? amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx))
                           : 0;
            const long_index_t a_n_offset =
                CTranspose ? 0
                           : amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));

            GridwiseGemm::template Run<HasMainKBlockLoop, InMemoryDataOperationEnum::Set>(
                p_as_grid + a_group_offset + a_n_offset,
                p_bs_grid + b_group_offset + b_n_offset,
                p_ds_grid_grp,
                p_e_grid + e_group_offset + e_n_offset,
                p_shared,
                a_element_op,
                b_element_op,
                cde_element_op,
                a_grid_desc_k0_m_k1,
                b_grid_desc_k0_n_k1,
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
                block_2_ctile_map);
        }
    }
#else
    ignore = p_as_grid;
    ignore = p_bs_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
    ignore = block_2_ctile_map;
#endif
}

} // namespace
#ifdef CK_CODE_GEN_RTC
template <typename T>
using is_tuple = decltype(ck::declval<T&>().IsTuple());
#else
template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());
#endif

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
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType,
          LoopScheduler LoopSched   = make_default_loop_scheduler(),
          index_t NumGroupsToMerge  = 1>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
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
    using DeviceOp = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle;
    GET_MXDL_PER_WAVE_IMPL
    // Force usage of 16x16 instruction for WMMA
    static constexpr bool Wave32Force16MNPerXDL =
        is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() &&
        sizeof(AComputeDataType) == 2 && sizeof(BComputeDataType) == 2 &&
        is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough> &&
        (ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ||
         ConvForwardSpecialization == ConvolutionForwardSpecialization::Default);
    static constexpr index_t Wave32MaxMNPerXDL =
        Wave32Force16MNPerXDL ? 16 : math::max(MPerXDL, NPerXDL);

    static constexpr auto MXdlPerWave64 = GetMXdlPerWave<true>();
    static constexpr auto MXdlPerWave32 =
        GetMXdlPerWave<false,
                       Wave32MaxMNPerXDL,
                       Wave32MaxMNPerXDL,
                       NXdlPerWave*(NPerXDL / Wave32MaxMNPerXDL)>();

    static_assert(NumGroupsToMerge >= 1);

    static constexpr bool isMultiA  = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB  = is_detected<is_tuple, BDataType>::value;
    static constexpr bool isMultiAB = isMultiA || isMultiB;

    // NGCHW is not supported for multiAB
    static_assert(!(is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                    is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()) ||
                  !(isMultiA || isMultiB));

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr bool DoElementwiseBeforeCShuffle =
        NumDTensor == 0 && !isMultiAB && is_same_v<EDataType, bhalf_t> &&
        !is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough>;

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

    using ConvToGemmFwdTransformer = TransformConvFwdToGemm<NDimSpatial,
                                                            ConvForwardSpecialization,
                                                            true /*SplitN*/,
                                                            ADataType,
                                                            EDataType,
                                                            NumGroupsToMerge,
                                                            index_t,
                                                            CTranspose>;

    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto conv_ngchw_to_nhwgc_transformer =
        TransformConvNGCHWToNHWGC<ALayout,
                                  BLayout,
                                  ELayout,
                                  NDimSpatial,
                                  NPerBlock / ClusterLengthNPerBlock,
                                  NPerBlock / ClusterLengthNPerBlock>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <typename ALay>
    static auto MakeAGridDescriptor_M_K(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
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

        return in_gemmm_gemmk_desc;
    }

    template <typename BLay>
    static auto MakeBGridDescriptor_N_K(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
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

        return wei_gemmn_gemmk_desc;
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

    // desc for problem definition
    constexpr static ConvToGemmFwdTransformer dummy_conv_to_gemm_transformer;
    using AGridDesc_M_K =
        remove_cvref_t<decltype(MakeAGridDescriptor_M_K<ALayout>(dummy_conv_to_gemm_transformer))>;
    using BGridDesc_N_K =
        remove_cvref_t<decltype(MakeBGridDescriptor_N_K<BLayout>(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;

    // If we are using multiAB and one of the template datatype parameters is not a tuple, convert
    // it to it
    using GemmADataType = std::conditional_t<!isMultiA && isMultiB, Tuple<ADataType>, ADataType>;
    using GemmBDataType = std::conditional_t<!isMultiB && isMultiA, Tuple<BDataType>, BDataType>;

#define CK_GRIDWISE_GEMM_FWD_MULTIPLE_ABD_XDL_CSHUFFLE_TEMPLATE_PARAMETERS                       \
    GemmADataType, GemmBDataType, AComputeDataType, AccDataType, CShuffleDataType, DsDataType,   \
        EDataType, AElementwiseOperation, BElementwiseOperation, CDEElementwiseOperation,        \
        InMemoryDataOperationEnum::Set, NumGemmKPrefetchStage, BlockSize, MPerBlock, NPerBlock,  \
        KPerBlock, AK1, BK1, MPerXDL_, NPerXDL_, MXdlPerWave_, NXdlPerWave*(NPerXDL / NPerXDL_), \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, ABlockTransferThreadClusterArrangeOrder,   \
        ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,                                \
        ABlockTransferSrcScalarPerVector, ABlockTransferDstScalarPerVector_AK1, false,           \
        ABlockLdsExtraM, BBlockTransferThreadClusterLengths_BK0_N_BK1,                           \
        BBlockTransferThreadClusterArrangeOrder, BBlockTransferSrcAccessOrder,                   \
        BBlockTransferSrcVectorDim, BBlockTransferSrcScalarPerVector,                            \
        BBlockTransferDstScalarPerVector_BK1, false, BBlockLdsExtraN,                            \
        CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle*(NPerXDL / NPerXDL_),       \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                        \
        CDEBlockTransferScalarPerVector_NPerBlock, LoopSched, PipelineVersion::v1,               \
        BComputeDataType

#define CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_XDL_CSHUFFLE_TEMPLATE_PARAMETERS                       \
    GemmADataType, GemmBDataType, AComputeDataType, AccDataType, CShuffleDataType, DsDataType, \
        EDataType, AElementwiseOperation, BElementwiseOperation, CDEElementwiseOperation,      \
        NumGemmKPrefetchStage, BlockSize, MPerBlock, NPerBlock, KPerBlock, AK1, BK1, MPerXDL_, \
        NPerXDL_, MXdlPerWave_, NXdlPerWave*(NPerXDL / NPerXDL_),                              \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, ABlockTransferThreadClusterArrangeOrder, \
        ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,                              \
        ABlockTransferSrcScalarPerVector, ABlockTransferDstScalarPerVector_AK1, false,         \
        ABlockLdsExtraM, BBlockTransferThreadClusterLengths_BK0_N_BK1,                         \
        BBlockTransferThreadClusterArrangeOrder, BBlockTransferSrcAccessOrder,                 \
        BBlockTransferSrcVectorDim, BBlockTransferSrcScalarPerVector,                          \
        BBlockTransferDstScalarPerVector_BK1, false, BBlockLdsExtraN,                          \
        CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle*(NPerXDL / NPerXDL_),     \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                      \
        CDEBlockTransferScalarPerVector_NPerBlock, LoopSched, PipelineVersion::v1,             \
        BComputeDataType, DoElementwiseBeforeCShuffle

#define CK_GRIDWISE_GEMM_FWD_CTRANSPOSE_XDL_CSHUFFLE_TEMPLATE_PARAMETERS                       \
    GemmBDataType, GemmADataType, AComputeDataType, AccDataType, CShuffleDataType, DsDataType, \
        EDataType, BElementwiseOperation, AElementwiseOperation, CDEElementwiseOperation,      \
        NumGemmKPrefetchStage, BlockSize, NPerBlock, MPerBlock, KPerBlock, BK1, AK1, NPerXDL,  \
        MPerXDL, NXdlPerWave, MXdlPerWave_, BBlockTransferThreadClusterLengths_BK0_N_BK1,      \
        BBlockTransferThreadClusterArrangeOrder, BBlockTransferSrcAccessOrder,                 \
        BBlockTransferSrcVectorDim, BBlockTransferSrcScalarPerVector,                          \
        BBlockTransferDstScalarPerVector_BK1, false, BBlockLdsExtraN,                          \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, ABlockTransferThreadClusterArrangeOrder, \
        ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,                              \
        ABlockTransferSrcScalarPerVector, ABlockTransferDstScalarPerVector_AK1, false,         \
        ABlockLdsExtraM, CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle,         \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                      \
        CDEBlockTransferScalarPerVector_NPerBlock, LoopSched, PipelineVersion::v1,             \
        BComputeDataType, DoElementwiseBeforeCShuffle

    // Use appropriate gridwise gemm
    template <index_t MXdlPerWave_, index_t MPerXDL_, index_t NPerXDL_>
    using GridwiseGemmMultipleABDBase = GridwiseGemmMultipleABD_xdl_cshuffle<
        CK_GRIDWISE_GEMM_FWD_MULTIPLE_ABD_XDL_CSHUFFLE_TEMPLATE_PARAMETERS>;
    template <index_t MXdlPerWave_, index_t MPerXDL_, index_t NPerXDL_>
    using GridwiseGemmMultipleDBase = GridwiseGemmMultipleD_xdl_cshuffle<
        CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_XDL_CSHUFFLE_TEMPLATE_PARAMETERS>;
    template <index_t MXdlPerWave_>
    using GridwiseGemmMultipleDCTransposeBase = GridwiseGemmMultipleD_xdl_cshuffle<
        CK_GRIDWISE_GEMM_FWD_CTRANSPOSE_XDL_CSHUFFLE_TEMPLATE_PARAMETERS>;
#undef CK_GRIDWISE_GEMM_FWD_MULTIPLE_ABD_XDL_CSHUFFLE_TEMPLATE_PARAMETERS
#undef CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_XDL_CSHUFFLE_TEMPLATE_PARAMETERS
#undef CK_GRIDWISE_GEMM_FWD_CTRANSPOSE_XDL_CSHUFFLE_TEMPLATE_PARAMETERS

    using GridwiseGemm64 = std::conditional_t<
        isMultiA || isMultiB,
        GridwiseGemmMultipleABDBase<math::max(MXdlPerWave64, 1), MPerXDL, NPerXDL>,
        GridwiseGemmMultipleDBase<math::max(MXdlPerWave64, 1), MPerXDL, NPerXDL>>;
    using GridwiseGemm32 = std::conditional_t<
        isMultiA || isMultiB,
        GridwiseGemmMultipleABDBase<MXdlPerWave32, Wave32MaxMNPerXDL, Wave32MaxMNPerXDL>,
        GridwiseGemmMultipleDBase<MXdlPerWave32, Wave32MaxMNPerXDL, Wave32MaxMNPerXDL>>;

    using GridwiseGemmCTranspose64 =
        std::conditional_t<CTranspose,
                           GridwiseGemmMultipleDCTransposeBase<math::max(MXdlPerWave64, 1)>,
                           GridwiseGemm64>;
    using GridwiseGemmCTranspose32 =
        std::conditional_t<CTranspose,
                           GridwiseGemmMultipleDCTransposeBase<MXdlPerWave32>,
                           GridwiseGemm32>;

    // If ADataTypes or BDataTypes is tuple, user has to pass std::array with pointers.
    using APointers =
        std::conditional_t<isMultiA, std::array<const void*, NumATensor>&, const void*>;
    using BPointers =
        std::conditional_t<isMultiB, std::array<const void*, NumBTensor>&, const void*>;
    // Use Tuple for the both cases for GridPointer to initialize it in Argument constructor (not
    // in initializer list what is required for single const pointer).
    using AGridPointer = remove_cvref_t<
        decltype(GetAGridPointer < isMultiA || isMultiB, GridwiseGemm64, ADataType > ())>;
    using BGridPointer = remove_cvref_t<
        decltype(GetBGridPointer < isMultiA || isMultiB, GridwiseGemm64, BDataType > ())>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(
            AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(
            BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemmCTranspose64::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemmCTranspose64::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemmCTranspose64::MakeDefaultBlock2ETileMap(
            EGridDesc_M_N{}))>;
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

    // NPerBlock is used for the first and second dim which to use
    // CDEBlockTransferScalarPerVector_NPerBlock for load and store during
    // transposition. CBlockTransferScalarPerVector_NWaveNPerXdl is aligned to
    // NPerBlock so it is more flexible to use this dim for load store dimension
    // with such scalar per vector.
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

    // Argument
    struct Argument : public BaseArgument
    {
        template <typename GridwiseGemm, typename GridwiseGemmCTranspose>
        void InitGridDesc()
        {
            // populate desc for Ds/E
            if constexpr(isMultiA || isMultiB)
            {
                const auto as_grid_desc_ak0_m_ak1 =
                    generate_tuple([&](auto) { return a_grid_desc_m_k_; }, Number<NumATensor>{});
                const auto bs_grid_desc_bk0_n_bk1 =
                    generate_tuple([&](auto) { return b_grid_desc_n_k_; }, Number<NumBTensor>{});

                if(GridwiseGemm::CheckValidity(as_grid_desc_ak0_m_ak1,
                                               bs_grid_desc_bk0_n_bk1,
                                               ds_grid_desc_m_n_,
                                               e_grid_desc_m_n_,
                                               block_2_etile_map_))
                {
                    e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            e_grid_desc_m_n_);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            ds_grid_desc_m_n_);
                }
            }
            else
            {
                bool valid = false;
                if constexpr(CTranspose)
                {
                    valid = GridwiseGemmCTranspose::CheckValidity(b_grid_desc_n_k_,
                                                                  a_grid_desc_m_k_,
                                                                  ds_grid_desc_m_n_,
                                                                  e_grid_desc_m_n_,
                                                                  block_2_etile_map_);
                }
                else
                {
                    valid = GridwiseGemmCTranspose::CheckValidity(a_grid_desc_m_k_,
                                                                  b_grid_desc_n_k_,
                                                                  ds_grid_desc_m_n_,
                                                                  e_grid_desc_m_n_,
                                                                  block_2_etile_map_);
                }
                if(valid)
                {
                    e_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmCTranspose::
                        MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n_);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmCTranspose::
                        MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n_);
                }
            }
        };

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
              p_ds_grid_{},
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
              a_grid_desc_m_k_{
                  DeviceOp::MakeAGridDescriptor_M_K<ALayout>(conv_to_gemm_transformer_)},
              b_grid_desc_n_k_{
                  DeviceOp::MakeBGridDescriptor_N_K<BLayout>(conv_to_gemm_transformer_)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N<ELayout>(conv_to_gemm_transformer_)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_etile_map_{
                  GridwiseGemmCTranspose64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              compute_ptr_offset_of_groups_{},
              compute_ptr_offset_of_n_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            // A/B/E Batch Stride
            if constexpr(isMultiA || isMultiB)
            {
                static_for<0, NumATensor, 1>{}([&](auto i) {
                    // Init compute_ptr_offset_of_groups_ for multiple AB
                    compute_ptr_offset_of_groups_.BatchStrideA_(i) =
                        a_g_n_c_wis_strides_[0] * NumGroupsToMerge;

                    // Use GemmADataType/GemmBDataType to iterate over tuple (even if passed data
                    // type is not tuple)
                    using DataType = remove_cvref_t<tuple_element_t<i.value, GemmADataType>>;
                    // It is possible that one of the AB is a pointer and one is a tuple.
                    // Then also use multiAB but we have to cast single pointer instead of tuple of
                    // pointer.
                    if constexpr(isMultiA)
                    {
                        // p_as is tuple
                        p_as_grid_(i) = static_cast<const DataType*>(p_as[i.value]);
                        // compute_ptr_offset_of_n_ not need BatchStrideB so
                        // in case of MultiA is false but isMultiB is true
                        // BatchStrideA_ is not tuple.
                        compute_ptr_offset_of_n_.BatchStrideA_(i) =
                            a_g_n_c_wis_strides_[1] * conv_N_per_block_;
                    }
                    else
                    {
                        // if MultiB and not MultiA then p_as is single pointer
                        p_as_grid_(i) = static_cast<const DataType*>(p_as);
                        compute_ptr_offset_of_n_.BatchStrideA_ =
                            a_g_n_c_wis_strides_[1] * conv_N_per_block_;
                    }
                });
                static_for<0, NumBTensor, 1>{}([&](auto i) {
                    // Init compute_ptr_offset_of_groups_ for multiple AB
                    compute_ptr_offset_of_groups_.BatchStrideB_(i) =
                        b_g_k_c_xs_strides_[0] * NumGroupsToMerge;

                    using DataType = remove_cvref_t<tuple_element_t<i.value, GemmBDataType>>;
                    // It is possible that one of the AB is a pointer and one is a tuple.
                    // Then also use multiAB but we have to cast single pointer instead of tuple of
                    // pointer.
                    if constexpr(isMultiB)
                    {
                        // p_bs is tuple
                        p_bs_grid_(i) = static_cast<const DataType*>(p_bs[i.value]);
                    }
                    else
                    {
                        // if MultiA and not MultiB then p_bs is single pointer
                        p_bs_grid_(i) = static_cast<const DataType*>(p_bs);
                    }
                });
            }
            else
            {
                compute_ptr_offset_of_groups_.BatchStrideA_ =
                    a_g_n_c_wis_strides_[0] * NumGroupsToMerge;
                compute_ptr_offset_of_groups_.BatchStrideB_ =
                    b_g_k_c_xs_strides_[0] * NumGroupsToMerge;
                compute_ptr_offset_of_n_.BatchStrideA_ =
                    a_g_n_c_wis_strides_[1] * conv_N_per_block_;

                // p_as and p_bs are pointers
                p_as_grid_(I0) = static_cast<const ADataType*>(p_as);
                p_bs_grid_(I0) = static_cast<const BDataType*>(p_bs);
            }

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);

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

            if(get_warp_size() == 64)
            {
                if constexpr(MXdlPerWave64 > 0)
                {
                    InitGridDesc<GridwiseGemm64, GridwiseGemmCTranspose64>();
                }
            }
            else
            {
                if constexpr(MXdlPerWave32 > 0)
                {
                    InitGridDesc<GridwiseGemm32, GridwiseGemmCTranspose32>();
                }
            }
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
            if constexpr(NeedTransposeKernel)
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

        void Print() const
        {
            std::cout << "AComputeDataType: " << get_type_name<AComputeDataType>()
                      << "; BComputeDataType: " << get_type_name<BComputeDataType>()
                      << "; EDataType: " << get_type_name<EDataType>() << std::endl;

            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;

            std::cout << "a grid desc" << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "b grid desc" << b_grid_desc_bk0_n_bk1_ << std::endl;
            std::cout << "e grid desc" << e_grid_desc_mblock_mperblock_nblock_nperblock_
                      << std::endl;
        }

        //  private:
        // pointers (tuple if multi AB, pointer if no)
        AGridPointer p_as_grid_;
        BGridPointer p_bs_grid_;
        typename GridwiseGemm64::DsGridPointer p_ds_grid_;
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

        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_b_, elementwise_block_2_ctile_map_transpose_e_;

        NGCHWTransposeDescType a_in_transpose_desc_, e_out_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, e_in_transpose_desc_;
        GKCYXTransposeDescType b_in_transpose_desc_;
        GKYXCTransposeDescType b_out_transpose_desc_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>
            compute_ptr_offset_of_groups_;
        ComputePtrOffsetOfStridedBatch<NumATensor, I1, NumDTensor> compute_ptr_offset_of_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm, typename GridwiseGemmCTranspose>
        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t num_workgroups_per_Conv_N =
                arg.a_g_n_c_wis_lengths_[I1] / arg.conv_N_per_block_;

            const index_t gdx = arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);
            const index_t gdy = arg.num_group_ / NumGroupsToMerge;
            const index_t gdz = num_workgroups_per_Conv_N;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                if constexpr(isMultiA || isMultiB)
                {
                    // Generate tuples with grid descriptors for each A and B
                    const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
                        [&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, Number<NumATensor>{});
                    const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
                        [&](auto) { return arg.b_grid_desc_bk0_n_bk1_; }, Number<NumBTensor>{});

                    const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                        GridwiseGemm,
                        AGridPointer,
                        BGridPointer,
                        typename GridwiseGemm::DsGridPointer,
                        EDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        decltype(as_grid_desc_ak0_m_ak1),
                        decltype(bs_grid_desc_bk0_n_bk1),
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        Block2ETileMap,
                        ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                        ComputePtrOffsetOfStridedBatch<NumATensor, I1, NumDTensor>,
                        has_main_loop,
                        isMultiA,
                        isMultiB,
                        CTranspose>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg.p_as_grid_,
                        arg.p_bs_grid_,
                        arg.p_ds_grid_,
                        arg.p_e_grid_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        as_grid_desc_ak0_m_ak1,
                        bs_grid_desc_bk0_n_bk1,
                        arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.block_2_etile_map_,
                        arg.compute_ptr_offset_of_groups_,
                        arg.compute_ptr_offset_of_n_);
                }
                else
                {
                    const ADataType* p_a_grid = arg.p_as_grid_.At(I0);
                    const BDataType* p_b_grid = arg.p_bs_grid_.At(I0);
                    EDataType* p_e_grid       = arg.p_e_grid_;
                    if constexpr(NeedTransposeKernel)
                    {
                        if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                                     is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
                        {
                            p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                            p_b_grid = type_convert<const BDataType*>(arg.p_workspace_) +
                                       arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);
                            p_e_grid = type_convert<EDataType*>(arg.p_workspace_) +
                                       (arg.GetWorkspaceATensorSizeBytes() +
                                        arg.GetWorkspaceBTensorSizeBytes()) /
                                           sizeof(EDataType);
                        }
                        else if constexpr(is_NGCHW_GKYXC_NGKHW<ALayout, BLayout, ELayout>() ||
                                          is_NGCDHW_GKZYXC_NGKDHW<ALayout, BLayout, ELayout>())
                        {
                            p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                            p_e_grid = type_convert<EDataType*>(arg.p_workspace_) +
                                       (arg.GetWorkspaceATensorSizeBytes() +
                                        arg.GetWorkspaceBTensorSizeBytes()) /
                                           sizeof(EDataType);
                        }
                    }

                    if constexpr(CTranspose)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                            GridwiseGemmCTranspose,
                            const BDataType*,
                            const ADataType*,
                            typename GridwiseGemm::DsGridPointer,
                            EDataType,
                            BElementwiseOperation,
                            AElementwiseOperation,
                            CDEElementwiseOperation,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            Block2ETileMap,
                            ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                            ComputePtrOffsetOfStridedBatch<NumATensor, I1, NumDTensor>,
                            has_main_loop,
                            isMultiA,
                            isMultiB,
                            CTranspose>;
                        return launch_and_time_kernel(
                            stream_config,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            p_b_grid,
                            p_a_grid,
                            arg.p_ds_grid_,
                            p_e_grid,
                            arg.b_element_op_,
                            arg.a_element_op_,
                            arg.cde_element_op_,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.block_2_etile_map_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                            GridwiseGemm,
                            const ADataType*,
                            const BDataType*,
                            typename GridwiseGemm::DsGridPointer,
                            EDataType,
                            AElementwiseOperation,
                            BElementwiseOperation,
                            CDEElementwiseOperation,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                            Block2ETileMap,
                            ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, NumDTensor>,
                            ComputePtrOffsetOfStridedBatch<NumATensor, I1, NumDTensor>,
                            has_main_loop,
                            isMultiA,
                            isMultiB,
                            CTranspose>;

                        return launch_and_time_kernel(
                            stream_config,
                            kernel,
                            dim3(gdx, gdy, gdz),
                            dim3(BlockSize),
                            0,
                            p_a_grid,
                            p_b_grid,
                            arg.p_ds_grid_,
                            p_e_grid,
                            arg.a_element_op_,
                            arg.b_element_op_,
                            arg.cde_element_op_,
                            arg.a_grid_desc_ak0_m_ak1_,
                            arg.b_grid_desc_bk0_n_bk1_,
                            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                            arg.block_2_etile_map_,
                            arg.compute_ptr_offset_of_groups_,
                            arg.compute_ptr_offset_of_n_);
                    }
                }
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        template <typename GridwiseGemm, typename GridwiseGemmCTranspose>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;
            if constexpr(NeedTransposeKernel)
            {
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
                BDataType* p_b_out_grid = type_convert<BDataType*>(arg.p_workspace_) +
                                          arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);

                auto kernel_transpose = kernel_elementwise_dual<GridwiseElementwiseInputTranspose,
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

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_transpose,
                                                   dim3(a_grid_size + b_grid_size),
                                                   dim3(ElementwiseBlocksize),
                                                   0,
                                                   make_tuple(arg.a_in_transpose_desc_),
                                                   make_tuple(arg.b_in_transpose_desc_),
                                                   make_tuple(arg.a_out_transpose_desc_),
                                                   make_tuple(arg.b_out_transpose_desc_),
                                                   make_tuple(arg.p_as_grid_.At(I0)),
                                                   make_tuple(arg.p_bs_grid_.At(I0)),
                                                   make_tuple(p_a_out_grid),
                                                   make_tuple(p_b_out_grid),
                                                   arg.elementwise_block_2_ctile_map_transpose_a_,
                                                   arg.elementwise_block_2_ctile_map_transpose_b_,
                                                   element_wise::PassThrough{},
                                                   a_grid_size);
            }

            avg_time += RunGemm<GridwiseGemm, GridwiseGemmCTranspose>(arg, stream_config);

            if constexpr(NeedTransposeKernel)
            {
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

                avg_time += launch_and_time_kernel(stream_config,
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

            return avg_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(get_warp_size() == 64)
            {
                if constexpr(MXdlPerWave64 > 0)
                {
                    return RunImp<GridwiseGemm64, GridwiseGemmCTranspose64>(arg, stream_config);
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                if constexpr(MXdlPerWave32 > 0)
                {
                    return RunImp<GridwiseGemm32, GridwiseGemmCTranspose32>(arg, stream_config);
                }
                else
                {
                    return 0;
                }
            }
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

        const index_t G                  = arg.b_g_k_c_xs_lengths_[I0];
        const index_t K                  = arg.b_g_k_c_xs_lengths_[I1];
        const index_t C                  = arg.b_g_k_c_xs_lengths_[I2];
        const index_t input_spatial_acum = ck::accumulate_n<index_t>(
            arg.a_g_n_c_wis_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

        // check device
        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }

        if(!ck::is_xdl_wmma_supported<AComputeDataType,
                                      BComputeDataType,
                                      Wave32MaxMNPerXDL,
                                      Wave32MaxMNPerXDL>())
        {
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
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter3x3)
        {
            if(C != 1)
            {
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = arg.b_g_k_c_xs_lengths_[i + I3];

                if(filter_spatial_dim != I3)
                {
                    return false;
                }
            }
        }

        if constexpr(NumGroupsToMerge > 1)
        {
            if(!(C == 1))
            {
                return false;
            }
            if(G % NumGroupsToMerge != 0)
            {
                return false;
            }
            if constexpr(!(is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() ||
                           is_NGCSpatial_GKSpatial_NGKSpatial<ALayout, BLayout, ELayout>() ||
                           is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                           is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()))
            {
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
            // Check access per C
            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                // If not possible, check access per G
                if(!(ABlockTransferSrcVectorDim == 1 && (C == 1 || NumGroupsToMerge == 1) &&
                     (is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() ||
                      is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                      is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>()) &&
                     G % ABlockTransferSrcScalarPerVector == 0))
                {
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
                    return false;
                }
                if(input_spatial_acum % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
        }
        else
        {
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
                return false;
            }
        }
        else
        {
            return false;
        }
        //  check vector access of Ds
        bool valid = true;

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
                    valid = false;
                }

                if constexpr(is_same_v<DLayout, ctc::G_K>)
                {
                    // G and K must be the same
                    if(arg.ds_g_n_k_wos_lengths_[i][0] != arg.e_g_n_k_wos_lengths_[0] ||
                       arg.ds_g_n_k_wos_lengths_[i][2] != arg.e_g_n_k_wos_lengths_[2])
                    {
                        valid = false;
                    }
                }
                else
                {
                    // E and D must have the same shape
                    for(index_t d = 0; d < NDimSpatial + 3; d++)
                    {
                        if(arg.ds_g_n_k_wos_lengths_[i][d] != arg.e_g_n_k_wos_lengths_[d])
                        {
                            valid = false;
                        }
                    }
                }
            }
            else
            {
                valid = false;
            }
        });

        if constexpr(NeedTransposeKernel)
        {
            if((G * C) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if((G * K) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                arg.e_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

            if(input_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if(output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if(!arg.p_workspace_)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Warning: Workspace for "
                                 "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument is not "
                                 "allocated, use SetWorkSpacePointer."
                              << std::endl;
                }
                return false;
            }

            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            if(!(arg.a_out_transpose_desc_.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
                 arg.e_in_transpose_desc_.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
            {
                return false;
            }
        }

        if(!valid)
        {
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK> || is_same_v<ELayout, ctc::NGKW> ||
                     is_same_v<ELayout, ctc::NGKHW> || is_same_v<ELayout, ctc::NGKDHW>)
        {
            if(CTranspose == false)
            {
                if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    return false;
                }
            }
            else
            {
                const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                    arg.e_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

                if(output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock != 0)
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }
        if constexpr(is_same_v<AComputeDataType, ck::tf32_t> ||
                     is_same_v<BComputeDataType, ck::tf32_t>)
        {
            if(!is_tf32_supported())
            {
                return false;
            }
            if constexpr(!is_same_v<AComputeDataType, BComputeDataType>)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "ComputeDataType for A and B should be same while using TF32"
                              << std::endl;
                }
                return false;
            }
        }
        // check Gridwise GEMM
        if(get_warp_size() == 64)
        {
            if constexpr(MXdlPerWave64 > 0)
            {
                if constexpr(isMultiA || isMultiB)
                {
                    // Genarate tuples with the same descriptors
                    const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
                        [&](auto) { return arg.a_grid_desc_m_k_; }, Number<NumATensor>{});
                    const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
                        [&](auto) { return arg.b_grid_desc_n_k_; }, Number<NumBTensor>{});
                    return GridwiseGemm64::CheckValidity(as_grid_desc_ak0_m_ak1,
                                                         bs_grid_desc_bk0_n_bk1,
                                                         arg.ds_grid_desc_m_n_,
                                                         arg.e_grid_desc_m_n_,
                                                         arg.block_2_etile_map_);
                }
                else
                {
                    if constexpr(CTranspose)
                    {
                        return GridwiseGemmCTranspose64::CheckValidity(arg.b_grid_desc_n_k_,
                                                                       arg.a_grid_desc_m_k_,
                                                                       arg.ds_grid_desc_m_n_,
                                                                       arg.e_grid_desc_m_n_,
                                                                       arg.block_2_etile_map_);
                    }
                    else
                    {
                        return GridwiseGemmCTranspose64::CheckValidity(arg.a_grid_desc_m_k_,
                                                                       arg.b_grid_desc_n_k_,
                                                                       arg.ds_grid_desc_m_n_,
                                                                       arg.e_grid_desc_m_n_,
                                                                       arg.block_2_etile_map_);
                    }
                }
            }
        }
        else
        {

            if constexpr(MXdlPerWave32 > 0)
            {
                if constexpr(isMultiA || isMultiB)
                {
                    // Genarate tuples with the same descriptors
                    const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
                        [&](auto) { return arg.a_grid_desc_m_k_; }, Number<NumATensor>{});
                    const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
                        [&](auto) { return arg.b_grid_desc_n_k_; }, Number<NumBTensor>{});
                    return GridwiseGemm32::CheckValidity(as_grid_desc_ak0_m_ak1,
                                                         bs_grid_desc_bk0_n_bk1,
                                                         arg.ds_grid_desc_m_n_,
                                                         arg.e_grid_desc_m_n_,
                                                         arg.block_2_etile_map_);
                }
                else
                {
                    if constexpr(CTranspose)
                    {
                        return GridwiseGemmCTranspose32::CheckValidity(arg.b_grid_desc_n_k_,
                                                                       arg.a_grid_desc_m_k_,
                                                                       arg.ds_grid_desc_m_n_,
                                                                       arg.e_grid_desc_m_n_,
                                                                       arg.block_2_etile_map_);
                    }
                    else
                    {
                        return GridwiseGemmCTranspose32::CheckValidity(arg.a_grid_desc_m_k_,
                                                                       arg.b_grid_desc_n_k_,
                                                                       arg.ds_grid_desc_m_n_,
                                                                       arg.e_grid_desc_m_n_,
                                                                       arg.block_2_etile_map_);
                    }
                }
            }
        }

        return false;
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

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle";

        if(get_warp_size() != 64) {
            str << "_WmmaPorted";
        }

        str    << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CDEBlockTransferScalarPerVector_NPerBlock << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << NumGroupsToMerge
            << ">";
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
                      "instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
        static_assert(ck_tile::reflect::conv::HasConvTraits<DeviceOp>,
                      "ConvTraits specialization not found for this device operation. "
                      "If you modified the template parameters of this class, ensure that "
                      "the corresponding ConvTraits specialization in "
                      "ck_tile/builder/reflect/conv_traits.hpp is updated to match, or that "
                      "InstanceTraits in "
                      "ck_tile/builder/reflect/"
                      "instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp "
                      "provides all required members for ConvTraits to work.");
        return std::make_unique<ck_tile::reflect::conv::ConvDescription>(
            ck_tile::reflect::describe<DeviceOp>());
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
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
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
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
