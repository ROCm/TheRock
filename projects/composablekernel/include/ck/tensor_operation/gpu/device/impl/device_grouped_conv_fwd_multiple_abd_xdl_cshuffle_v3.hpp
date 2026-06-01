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
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/host_utility/io.hpp"
#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/conv_describe.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
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
          typename ComputePtrOffset,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename DsGridDesc_M_N,
          typename EGridDesc_M_N,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_fwd_xdl_cshuffle_v3(typename GridwiseGemm::Argument karg,
                                            const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                                            const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                                            const DsGridDesc_M_N ds_grid_desc_m_n,
                                            const EGridDesc_M_N c_grid_desc_m_n,
                                            const ComputePtrOffset compute_ptr_offset_of_groups,
                                            const ComputePtrOffset compute_ptr_offset_of_n)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // offset base pointer for each work-group
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);
        const auto& ds_n_offset     = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);

        static constexpr index_t NumDTensor = GridwiseGemm::NumDTensor;
        using DsGridPointer                 = typename GridwiseGemm::DsGridPointer;
        DsGridPointer p_ds_grid_grp{};

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_n_offset[i] + ds_group_offset[i];
        });

        const long_index_t a_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
        const long_index_t b_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
        const long_index_t e_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

        const long_index_t a_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        using Block2CTileMap         = typename GridwiseGemm::Block2CTileMapDefault;
        const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

        if constexpr(GridwiseGemm::DirectLoadEnabled)
        {
#if defined(__gfx950__)
            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                karg.p_a_grid + a_group_offset + a_n_offset,
                karg.p_b_grid + b_group_offset,
                p_ds_grid_grp,
                karg.p_c_grid + e_group_offset + e_n_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.c_element_op,
                block_2_ctile_map,
                GridwiseGemm::template TransformGrid<decltype(a_grid_desc_ak0_m_ak1),
                                                     GridwiseGemm::AK0Number,
                                                     GridwiseGemm::AK1Number>(
                    a_grid_desc_ak0_m_ak1),
                GridwiseGemm::template TransformGrid<decltype(b_grid_desc_bk0_n_bk1),
                                                     GridwiseGemm::BK0Number,
                                                     GridwiseGemm::BK1Number>(
                    b_grid_desc_bk0_n_bk1),
                ds_grid_desc_m_n,
                c_grid_desc_m_n);
#endif
        }
        else
        {
            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                karg.p_a_grid + a_group_offset + a_n_offset,
                karg.p_b_grid + b_group_offset,
                p_ds_grid_grp,
                karg.p_c_grid + e_group_offset + e_n_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.c_element_op,
                block_2_ctile_map,
                GridwiseGemm::template TransformGrid<decltype(a_grid_desc_ak0_m_ak1),
                                                     GridwiseGemm::AK0Number,
                                                     GridwiseGemm::AK1Number>(
                    a_grid_desc_ak0_m_ak1),
                GridwiseGemm::template TransformGrid<decltype(b_grid_desc_bk0_n_bk1),
                                                     GridwiseGemm::BK0Number,
                                                     GridwiseGemm::BK1Number>(
                    b_grid_desc_bk0_n_bk1),
                ds_grid_desc_m_n,
                c_grid_desc_m_n);
        }
    }
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_m_n;
    ignore = c_grid_desc_m_n;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          typename ComputePtrOffset,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename DsGridDesc_M_N,
          typename EGridDesc_M_N,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        const DsGridDesc_M_N ds_grid_desc_m_n,
        const EGridDesc_M_N c_grid_desc_m_n,
        const ComputePtrOffset compute_ptr_offset_of_groups,
        const ComputePtrOffset compute_ptr_offset_of_n)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // offset base pointer for each work-group
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);
        const auto& ds_n_offset     = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);

        static constexpr index_t NumDTensor = GridwiseGemm::NumDTensor;
        using DsGridPointer                 = typename GridwiseGemm::DsGridPointer;
        DsGridPointer p_ds_grid_grp{};

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_n_offset[i] + ds_group_offset[i];
        });

        const long_index_t a_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
        const long_index_t b_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
        const long_index_t e_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

        const long_index_t a_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        using Block2CTileMap         = typename GridwiseGemm::Block2CTileMapDefault;
        const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

        if constexpr(GridwiseGemm::DirectLoadEnabled)
        {
#if defined(__gfx950__)
            GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                karg.p_a_grid + a_group_offset + a_n_offset,
                karg.p_b_grid + b_group_offset,
                p_ds_grid_grp,
                karg.p_c_grid + e_group_offset + e_n_offset,
                p_shared_0,
                p_shared_1,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.c_element_op,
                block_2_ctile_map,
                GridwiseGemm::template TransformGrid<decltype(a_grid_desc_ak0_m_ak1),
                                                     GridwiseGemm::AK0Number,
                                                     GridwiseGemm::AK1Number>(
                    a_grid_desc_ak0_m_ak1),
                GridwiseGemm::template TransformGrid<decltype(b_grid_desc_bk0_n_bk1),
                                                     GridwiseGemm::BK0Number,
                                                     GridwiseGemm::BK1Number>(
                    b_grid_desc_bk0_n_bk1),
                ds_grid_desc_m_n,
                c_grid_desc_m_n);
#endif
        }
        else
        {
            GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                karg.p_a_grid + a_group_offset + a_n_offset,
                karg.p_b_grid + b_group_offset,
                p_ds_grid_grp,
                karg.p_c_grid + e_group_offset + e_n_offset,
                p_shared_0,
                p_shared_1,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.c_element_op,
                block_2_ctile_map,
                GridwiseGemm::template TransformGrid<decltype(a_grid_desc_ak0_m_ak1),
                                                     GridwiseGemm::AK0Number,
                                                     GridwiseGemm::AK1Number>(
                    a_grid_desc_ak0_m_ak1),
                GridwiseGemm::template TransformGrid<decltype(b_grid_desc_bk0_n_bk1),
                                                     GridwiseGemm::BK0Number,
                                                     GridwiseGemm::BK1Number>(
                    b_grid_desc_bk0_n_bk1),
                ds_grid_desc_m_n,
                c_grid_desc_m_n);
        }
    }
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_m_n;
    ignore = c_grid_desc_m_n;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif // end of if (defined(__gfx9__))
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
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType,
          bool DirectLoad           = false>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
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
    using DeviceOp = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3;
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

    static constexpr bool isMultiA   = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB   = is_detected<is_tuple, BDataType>::value;
    static constexpr bool isMultiD   = DsDataType::Size() > 0;
    static constexpr bool isMultiABD = isMultiA && isMultiB && isMultiD;

    static constexpr bool DoElementwiseBeforeCShuffle =
        !isMultiD && is_same_v<EDataType, bhalf_t> &&
        !is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough>;

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // Generate vector size for C & Ds
    using CDEBlockTransferScalarPerVectors =
        typename uniform_sequence_gen<NumDTensor + 1,
                                      CDEBlockTransferScalarPerVector_NPerBlock>::type;

    using ConvToGemmFwdTransformer = TransformConvFwdToGemm<NDimSpatial,
                                                            ConvForwardSpecialization,
                                                            true /*SplitN*/,
                                                            ADataType,
                                                            EDataType>;

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
             is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
             ctc::NHWGC,
             std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
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
             is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
             ctc::GKYXC,
             std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
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
             is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
             ctc::NHWGK,
             std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
                                ctc::NDHWGK,
                                ELay>>;

        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<Layout>();

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
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
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;

    static constexpr index_t ABlockTransferSrcScalarPerVectorAligned =
        ABlockTransferSrcScalarPerVector * sizeof(ADataType) == 8
            ? 4 / sizeof(ADataType)
            : ABlockTransferSrcScalarPerVector;
    static constexpr index_t BBlockTransferSrcScalarPerVectorAligned =
        BBlockTransferSrcScalarPerVector * sizeof(BDataType) == 8
            ? 4 / sizeof(BDataType)
            : BBlockTransferSrcScalarPerVector;

    // Use appropriate gridwise gemm
    template <index_t MXdlPerWave_, index_t MPerXDL_, index_t NPerXDL_>
    using GridwiseGemmBase = GridwiseGemmMultiD_xdl_cshuffle_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        DsLayout,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
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
        MPerXDL_,
        NPerXDL_,
        MXdlPerWave_,
        NXdlPerWave*(NPerXDL / NPerXDL_),
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        DirectLoad ? ABlockTransferSrcScalarPerVectorAligned : ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        DirectLoad ? BBlockTransferSrcScalarPerVectorAligned : BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle*(NPerXDL / NPerXDL_),
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        AComputeDataType,
        BComputeDataType,
        ADataType,
        BDataType,
        DoElementwiseBeforeCShuffle,
        DirectLoad>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(MXdlPerWave64, 1), MPerXDL, NPerXDL>;
    using GridwiseGemm32 = GridwiseGemmBase<MXdlPerWave32, Wave32MaxMNPerXDL, Wave32MaxMNPerXDL>;

    // #undef GridwiseGemmV3TemplateParams

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
        Argument(const void* p_as,
                 const void* p_bs,
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
            : p_a_grid_{},
              p_b_grid_{},
              p_ds_grid_{p_ds},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                  a_g_n_c_wis_lengths, a_g_n_c_wis_strides)},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{conv_ngchw_to_nhwgc_transformer.TransposeWeiStrides(
                  b_g_k_c_xs_lengths, b_g_k_c_xs_strides)},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                  e_g_n_k_wos_lengths, e_g_n_k_wos_strides)},
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
            compute_ptr_offset_of_groups_.BatchStrideA_ = a_g_n_c_wis_strides_[0];
            compute_ptr_offset_of_groups_.BatchStrideB_ = b_g_k_c_xs_strides_[0];
            compute_ptr_offset_of_n_.BatchStrideA_ = a_g_n_c_wis_strides_[1] * conv_N_per_block_;

            // p_as and p_bs are pointers
            p_a_grid_ = static_cast<const ADataType*>(p_as);
            p_b_grid_ = static_cast<const BDataType*>(p_bs);

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                // D batch stride
                compute_ptr_offset_of_groups_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_[i][0];
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

            compute_ptr_offset_of_groups_.BatchStrideE_ = e_g_n_k_wos_strides_[0];
            compute_ptr_offset_of_n_.BatchStrideE_ = e_g_n_k_wos_strides_[1] * conv_N_per_block_;

            if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
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
            if constexpr(is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>())
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
            if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
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
            if constexpr(is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>())
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
            std::cout << "A[AK0, M, AK1]: " << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "B[BK0, N, BK1]: " << b_grid_desc_bk0_n_bk1_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers (tuple if multi AB, pointer if no)
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
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

        template <typename GridwiseGemm>
        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
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
            // TODO: Do we want to support kbatch ??
            std::tie(gdx, gdy, gdz) =
                GridwiseGemm::CalculateGridSize(GemmM, GemmN, I1 /*arg.KBatch*/);

            gdy = arg.num_group_;
            gdz = num_workgroups_per_Conv_N;

            index_t K_split                  = (GemmK + KPerBlock - 1) / KPerBlock * KPerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;
            EDataType* p_e_grid       = arg.p_e_grid_;

            if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                         is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
            {
                p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                p_b_grid = type_convert<const BDataType*>(arg.p_workspace_) +
                           arg.GetWorkspaceATensorSizeBytes() / sizeof(BDataType);
                p_e_grid =
                    type_convert<EDataType*>(arg.p_workspace_) +
                    (arg.GetWorkspaceATensorSizeBytes() + arg.GetWorkspaceBTensorSizeBytes()) /
                        sizeof(EDataType);
            }

            typename GridwiseGemm::Argument gemm_arg{
                p_a_grid,
                p_b_grid,
                arg.p_ds_grid_,
                p_e_grid,
                GemmM,
                GemmN,
                GemmK,
                // No need to set strides, we pass descs to kernel
                I0,
                I0,
                {},
                I0,
                I1, // kbatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.cde_element_op_};

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
                    };

                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_ak0_m_ak1_,
                        arg.b_grid_desc_bk0_n_bk1_,
                        arg.ds_grid_desc_m_n_,
                        arg.e_grid_desc_m_n_,
                        arg.compute_ptr_offset_of_groups_,
                        arg.compute_ptr_offset_of_n_);
                }
                else
                {
                    ave_time += launch_and_time_kernel(stream_config,
                                                       kernel,
                                                       dim3(gdx, gdy, gdz),
                                                       dim3(BlockSize),
                                                       0,
                                                       gemm_arg,
                                                       arg.a_grid_desc_ak0_m_ak1_,
                                                       arg.b_grid_desc_bk0_n_bk1_,
                                                       arg.ds_grid_desc_m_n_,
                                                       arg.e_grid_desc_m_n_,
                                                       arg.compute_ptr_offset_of_groups_,
                                                       arg.compute_ptr_offset_of_n_);
                }
            };

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    const auto kernel =
                        kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                ComputePtrOffset,
                                                                DeviceOp::AGridDesc_AK0_M_AK1,
                                                                DeviceOp::BGridDesc_BK0_N_BK1,
                                                                DeviceOp::DsGridDesc_M_N,
                                                                DeviceOp::EGridDesc_M_N,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                    Run(kernel);
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::One>;
                        Run(kernel);
                    }
                    else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Full)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
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
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Two>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Three)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Three>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Four)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Four>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Five)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
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
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Six>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Seven)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Seven>;
                            Run(kernel);
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            ComputePtrOffset,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::DsGridDesc_M_N,
                            DeviceOp::EGridDesc_M_N,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            ComputePtrOffset,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::DsGridDesc_M_N,
                            DeviceOp::EGridDesc_M_N,
                            true,
                            InMemoryDataOperationEnum::Set,
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
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Even>;
                        Run(kernel);
                    }
                }
            }
            // has_main_k_block_loop
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    const auto kernel =
                        kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                ComputePtrOffset,
                                                                DeviceOp::AGridDesc_AK0_M_AK1,
                                                                DeviceOp::BGridDesc_BK0_N_BK1,
                                                                DeviceOp::DsGridDesc_M_N,
                                                                DeviceOp::EGridDesc_M_N,
                                                                false,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                    Run(kernel);
                }
            }

            return ave_time;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;
            if constexpr(!isMultiABD)
            {
                // Transpose to NGHWC layotu
                if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                             is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
                {
                    const index_t a_grid_size =
                        arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                            arg.a_in_transpose_desc_);
                    const index_t b_grid_size =
                        arg.elementwise_block_2_ctile_map_transpose_b_.CalculateGridSize(
                            arg.b_in_transpose_desc_);

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

                    avg_time +=
                        launch_and_time_kernel(stream_config,
                                               kernel_transpose,
                                               dim3(a_grid_size + b_grid_size),
                                               dim3(ElementwiseBlocksize),
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
                                               a_grid_size);
                }

                avg_time += RunGemm<GridwiseGemm>(arg, stream_config);

                // Transpose result back to NGCHW
                if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                             is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
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

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(get_warp_size() == 64)
            {
                if constexpr(MXdlPerWave64 > 0)
                {
                    return RunImp<GridwiseGemm64>(arg, stream_config);
                }
            }
            else
            {
                if constexpr(MXdlPerWave32 > 0)
                {
                    return RunImp<GridwiseGemm32>(arg, stream_config);
                }
            }
            return 0;
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
        // Move this to runtime check to align Conv instances
        // with Conv Multiple D instances
        if constexpr(isMultiABD)
        {
            return false;
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The MultiABD is not supported!" << " In " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
        }

        // check device
        if constexpr(DirectLoad)
        {
            if(get_device_name() != "gfx950")
            {
                return false;
            }
        }

        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout
                        << "On gfx908 the accumulation data type must be one of fp32 or int32!"
                        << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                        << std::endl;
                }
                return false;
            }
        }

        if(!ck::is_xdl_wmma_supported<AComputeDataType,
                                      BComputeDataType,
                                      Wave32MaxMNPerXDL,
                                      Wave32MaxMNPerXDL>())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Current device does not support xdl instructions!" << " In "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
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
                        std::cout << "The input paramters do not align with specialization "
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
                        std::cout
                            << "The input paramters do not align with specialization Filter1x1Pad0!"
                            << " In " << __FILE__ << ":" << __LINE__
                            << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC> || is_same_v<ALayout, ctc::NGCW> ||
                     is_same_v<ALayout, ctc::NGCHW> || is_same_v<ALayout, ctc::NGCDHW>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
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
                std::cout << "Unsupported A Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if constexpr(is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                     is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
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

            const index_t input_spatial_acum = ck::accumulate_n<index_t>(
                arg.a_g_n_c_wis_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());
            const index_t output_spatial_acum = ck::accumulate_n<index_t>(
                arg.e_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

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
                    std::cout << "Warning: Workspace for "
                                 "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3::Argument is not "
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

        if(get_warp_size() == 64)
        {
            if constexpr(MXdlPerWave64 > 0)
            {
                typename GridwiseGemm64::Argument gemm_arg{nullptr,
                                                           nullptr,
                                                           {},
                                                           nullptr,
                                                           GemmM,
                                                           GemmN,
                                                           GemmK,
                                                           I0,
                                                           I0,
                                                           {},
                                                           I0,
                                                           I1 /*KBatch*/,
                                                           arg.a_element_op_,
                                                           arg.b_element_op_,
                                                           arg.cde_element_op_};
                return GridwiseGemm64::CheckValidity(gemm_arg);
            }
        }
        else
        {
            if constexpr(MXdlPerWave32 > 0)
            {
                typename GridwiseGemm32::Argument gemm_arg{nullptr,
                                                           nullptr,
                                                           {},
                                                           nullptr,
                                                           GemmM,
                                                           GemmN,
                                                           GemmK,
                                                           I0,
                                                           I0,
                                                           {},
                                                           I0,
                                                           I1 /*KBatch*/,
                                                           arg.a_element_op_,
                                                           arg.b_element_op_,
                                                           arg.cde_element_op_};
                return GridwiseGemm32::CheckValidity(gemm_arg);
            }
        }

        return false;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_as,
        const void* p_bs,
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
    MakeArgument(const void* p_as,
                 const void* p_bs,
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
        const void* p_a,
        const void* p_b,
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
        return std::make_unique<Argument>(p_a,
                                          p_b,
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
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
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

        return std::make_unique<Argument>(p_a,
                                          p_b,
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
        str << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3";

        if(get_warp_size() != 64) {
            str << "_WmmaPorted";
        }

        if constexpr(DirectLoad) {
            str << "_DirectLoad";
        }

        str << "<"
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
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer]
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
                      "instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
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
