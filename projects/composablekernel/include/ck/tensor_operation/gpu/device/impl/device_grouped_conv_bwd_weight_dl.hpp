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
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_dl.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_B_K0_M0_M1_K1,
          typename BGridDesc_B_K0_N0_N1_K1,
          typename CGridDesc_M0_M10_M11_N0_N10_N11,
          typename Block2CTileMap,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_gemm_dlops_bwd_weight(
        const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        const index_t batch_count,
        const AGridDesc_B_K0_M0_M1_K1 a_grid_desc_kbatch_k0_m0_m1_k1,
        const BGridDesc_B_K0_N0_N1_K1 b_grid_desc_kbatch_k0_n0_n1_k1,
        const CGridDesc_M0_M10_M11_N0_N10_N11 c_grid_desc_m0_m10_m11_n0_n10_n11,
        const Block2CTileMap block_2_ctile_map,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch)
{
#if(defined(__gfx906__) || defined(__gfx103__) || defined(__gfx90a__) || defined(__gfx908__) || \
    defined(__gfx94__) || defined(__gfx101__) || defined(__gfx11__) || defined(__gfx12__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t c_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

    __shared__ FloatAB p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB)];

    GridwiseGemm::template Run<HasMainKBlockLoop, HasDoubleTailKBlockLoop>(
        p_a_grid + a_batch_offset,
        p_b_grid + b_batch_offset,
        p_c_grid + c_batch_offset,
        p_shared,
        a_grid_desc_kbatch_k0_m0_m1_k1,
        b_grid_desc_kbatch_k0_n0_n1_k1,
        c_grid_desc_m0_m10_m11_n0_n10_n11,
        block_2_ctile_map,
        integral_constant<bool, HasMainKBlockLoop>{},
        integral_constant<bool, HasDoubleTailKBlockLoop>{});
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = batch_count;
    ignore = a_grid_desc_kbatch_k0_m0_m1_k1;
    ignore = b_grid_desc_kbatch_k0_n0_n1_k1;
    ignore = c_grid_desc_m0_m10_m11_n0_n10_n11;
    ignore = block_2_ctile_map;
    ignore = compute_ptr_offset_of_batch;
#endif
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
          ck::index_t K0PerBlock,
          ck::index_t K1,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          typename M1N1ThreadClusterM1Xs,
          typename M1N1ThreadClusterN1Xs,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct DeviceGroupedConvBwdWeight_Dl : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                                                         InLayout,
                                                                         WeiLayout,
                                                                         OutLayout,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementwiseOperation,
                                                                         WeiElementwiseOperation,
                                                                         OutElementwiseOperation>
{
    using DeviceOp = DeviceGroupedConvBwdWeight_Dl;

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

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

    static constexpr auto spatial_offset = I3;

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

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
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        const ck::index_t batch_k)
    {
        using namespace ck;

        const index_t N             = a_g_n_c_wis_lengths[I1];
        const index_t K             = b_g_k_c_xs_lengths[I1];
        const index_t C             = a_g_n_c_wis_lengths[I2];
        const index_t Wi            = a_g_n_c_wis_lengths[spatial_offset];
        const index_t Wo            = e_g_n_k_wos_lengths[spatial_offset];
        const index_t X             = b_g_k_c_xs_lengths[spatial_offset];
        const index_t InLeftPadW    = input_left_pads[I0];
        const index_t InRightPadW   = input_right_pads[I0];
        const index_t ConvStrideW   = conv_filter_strides[I0];
        const index_t ConvDilationW = conv_filter_dilations[I0];

        const auto InNStride  = a_g_n_c_wis_strides[I1];
        const auto InCStride  = a_g_n_c_wis_strides[I2];
        const auto InWStride  = a_g_n_c_wis_strides[spatial_offset];
        const auto WeiKStride = b_g_k_c_xs_strides[I1];
        const auto WeiCStride = b_g_k_c_xs_strides[I2];
        const auto OutKStride = e_g_n_k_wos_strides[I2];
        const auto OutWStride = e_g_n_k_wos_strides[spatial_offset];

        const index_t GemmKTotal = N * Wo;
        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Wo, K), make_tuple(OutWStride, OutKStride));

            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Wi, C), make_tuple(InWStride, InCStride));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weights tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Wo, K), make_tuple(OutWStride, OutKStride));
            const auto in_n_wi_c_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(InNStride, InWStride, InCStride));

            // A: output tensor
            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(X, C)),
                                                       make_merge_transform(make_tuple(N, Wo))),
                                            make_tuple(Sequence<1, 3>{}, Sequence<0, 2>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }

    } // function end
    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        const ck::index_t batch_k)
    {
        using namespace ck;

        const index_t N  = a_g_n_c_wis_lengths[I1];
        const index_t K  = b_g_k_c_xs_lengths[I1];
        const index_t C  = a_g_n_c_wis_lengths[I2];
        const index_t Hi = a_g_n_c_wis_lengths[spatial_offset];
        const index_t Wi = a_g_n_c_wis_lengths[spatial_offset + I1];
        const index_t Ho = e_g_n_k_wos_lengths[spatial_offset];
        const index_t Wo = e_g_n_k_wos_lengths[spatial_offset + I1];
        const index_t Y  = b_g_k_c_xs_lengths[spatial_offset];
        const index_t X  = b_g_k_c_xs_lengths[spatial_offset + I1];

        const index_t InLeftPadH    = input_left_pads[I0];
        const index_t InLeftPadW    = input_left_pads[I1];
        const index_t InRightPadH   = input_right_pads[I0];
        const index_t InRightPadW   = input_right_pads[I1];
        const index_t ConvStrideH   = conv_filter_strides[I0];
        const index_t ConvStrideW   = conv_filter_strides[I1];
        const index_t ConvDilationH = conv_filter_dilations[I0];
        const index_t ConvDilationW = conv_filter_dilations[I1];

        const auto InNStride  = a_g_n_c_wis_strides[I1];
        const auto InCStride  = a_g_n_c_wis_strides[I2];
        const auto InHStride  = a_g_n_c_wis_strides[spatial_offset];
        const auto InWStride  = a_g_n_c_wis_strides[spatial_offset + I1];
        const auto WeiKStride = b_g_k_c_xs_strides[I1];
        const auto WeiCStride = b_g_k_c_xs_strides[I2];
        const auto OutKStride = e_g_n_k_wos_strides[I2];
        const auto OutWStride = e_g_n_k_wos_strides[spatial_offset + I1];

        const index_t GemmKTotal = N * Ho * Wo;
        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Ho * Wo, K), make_tuple(OutWStride, OutKStride));

            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Hi * Wi, C), make_tuple(InWStride, InCStride));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, Y * X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Ho * Wo, K), make_tuple(OutWStride, OutKStride));
            const auto in_n_hi_wi_c_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(InNStride, InHStride, InWStride, InCStride));

            // A: output tensor
            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, Y * X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }

    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        const ck::index_t batch_k)
    {
        using namespace ck;

        const index_t N  = a_g_n_c_wis_lengths[I1];
        const index_t K  = b_g_k_c_xs_lengths[I1];
        const index_t C  = a_g_n_c_wis_lengths[I2];
        const index_t Di = a_g_n_c_wis_lengths[spatial_offset + I0];
        const index_t Hi = a_g_n_c_wis_lengths[spatial_offset + I1];
        const index_t Wi = a_g_n_c_wis_lengths[spatial_offset + I2];
        const index_t Do = e_g_n_k_wos_lengths[spatial_offset + I0];
        const index_t Ho = e_g_n_k_wos_lengths[spatial_offset + I1];
        const index_t Wo = e_g_n_k_wos_lengths[spatial_offset + I2];
        const index_t Z  = b_g_k_c_xs_lengths[spatial_offset + I0];
        const index_t Y  = b_g_k_c_xs_lengths[spatial_offset + I1];
        const index_t X  = b_g_k_c_xs_lengths[spatial_offset + I2];

        const index_t InLeftPadD    = input_left_pads[I0];
        const index_t InLeftPadH    = input_left_pads[I1];
        const index_t InLeftPadW    = input_left_pads[I2];
        const index_t InRightPadD   = input_right_pads[I0];
        const index_t InRightPadH   = input_right_pads[I1];
        const index_t InRightPadW   = input_right_pads[I2];
        const index_t ConvStrideD   = conv_filter_strides[I0];
        const index_t ConvStrideH   = conv_filter_strides[I1];
        const index_t ConvStrideW   = conv_filter_strides[I2];
        const index_t ConvDilationD = conv_filter_dilations[I0];
        const index_t ConvDilationH = conv_filter_dilations[I1];
        const index_t ConvDilationW = conv_filter_dilations[I2];

        const auto InNStride  = a_g_n_c_wis_strides[I1];
        const auto InCStride  = a_g_n_c_wis_strides[I2];
        const auto InDStride  = a_g_n_c_wis_strides[spatial_offset];
        const auto InHStride  = a_g_n_c_wis_strides[spatial_offset + I1];
        const auto InWStride  = a_g_n_c_wis_strides[spatial_offset + I2];
        const auto WeiKStride = b_g_k_c_xs_strides[I1];
        const auto WeiCStride = b_g_k_c_xs_strides[I2];
        const auto OutKStride = e_g_n_k_wos_strides[I2];
        const auto OutWStride = e_g_n_k_wos_strides[spatial_offset + I2];

        const index_t GemmKTotal = N * Do * Ho * Wo;
        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Do * Ho * Wo, K), make_tuple(OutWStride, OutKStride));

            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Di * Hi * Wi, C), make_tuple(InWStride, InCStride));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, Z * Y * X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * Do * Ho * Wo, K), make_tuple(OutWStride, OutKStride));
            const auto in_n_di_hi_wi_c_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(InNStride, InDStride, InHStride, InWStride, InCStride));

            // A: output tensor
            const auto out_gemmkpad_gemmmpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmktotal_gemmm_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, MPerBlock),
                    Sequence<true, true>{});

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmmpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(out_gemmkpad_gemmmpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_dip_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    in_gemmktotal_gemmn_grid_desc,
                    make_tuple(GemmK1Number * K0PerBlock * GemmKBatch, NPerBlock),
                    Sequence<true, true>{});

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmnpad_grid_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(GemmKBatch, GemmK0, GemmK1Number)),
                    make_pass_through_transform(in_gemmkpad_gemmnpad_grid_desc.GetLength(I1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(K, Z * Y * X * C), make_tuple(WeiKStride, WeiCStride));

            const auto wei_gemmmpad_gemmnpad_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(wei_gemmm_gemmn_grid_desc,
                                                                  make_tuple(MPerBlock, NPerBlock),
                                                                  Sequence<true, true>{});

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmmpad_gemmnpad_grid_desc);
        }

    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, 1);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 1);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>({1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  {1, 1, 1},
                                                                  1);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_B_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_B_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N       = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemm =
        GridwiseGemmDl_bkm_bkn_mn_v1r3<BlockSize,
                                       ADataType,
                                       AccDataType,
                                       CDataType,
                                       InMemoryDataOperationEnum::Set,
                                       AGridDesc_B_K0_M_K1,
                                       BGridDesc_B_K0_N_K1,
                                       CGridDesc_M_N,
                                       MPerBlock,
                                       NPerBlock,
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

    // Argument
    using AGridDesc_B_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_B_K0_M0_M1_K1(AGridDesc_B_K0_M_K1{}));
    using BGridDesc_B_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_B_K0_N0_N1_K1(BGridDesc_B_K0_N_K1{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{}));
    using Block2CTileMap =
        decltype(GridwiseGemm::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    struct Argument : public BaseArgument, public ArgumentSplitK
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
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
              a_element_op_{out_element_op},
              b_element_op_{wei_element_op},
              c_element_op_{in_element_op},
              Conv_G_{a_g_n_c_wis_lengths[I0]},
              Conv_K_{b_g_k_c_xs_lengths[I1]},
              Conv_C_{a_g_n_c_wis_lengths[I2]},
              filter_lengths_{b_g_k_c_xs_lengths},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            k_batch_ = split_k;

            const auto descs =
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                    a_g_n_c_wis_lengths, // input
                    a_g_n_c_wis_strides,
                    b_g_k_c_xs_lengths, // weight
                    b_g_k_c_xs_strides,
                    e_g_n_k_wos_lengths, // output
                    e_g_n_k_wos_strides,
                    conv_filter_strides,
                    conv_filter_dilations,
                    input_left_pads,
                    input_right_pads,
                    k_batch_);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            a_grid_desc_kbatch_k0_m0_m1_k1_ =
                GridwiseGemm::MakeAGridDescriptor_B_K0_M0_M1_K1(a_grid_desc_kbatch_k0_m_k1_);
            b_grid_desc_kbatch_k0_n0_n1_k1_ =
                GridwiseGemm::MakeBGridDescriptor_B_K0_N0_N1_K1(b_grid_desc_kbatch_k0_n_k1_);
            c_grid_desc_m0_m10_m11_n0_n10_n11_ =
                GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n_);
            ck::index_t M01 = 1;
            ck::index_t N01 = 1;
            block_2_ctile_map_ =
                GridwiseGemm::MakeCBlockClusterAdaptor(c_grid_desc_m_n_, M01, N01, k_batch_);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = e_g_n_k_wos_strides[I0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = a_g_n_c_wis_strides[I0];
            compute_ptr_offset_of_batch_.BatchStrideC_ = b_g_k_c_xs_strides[I0];
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;

        AGridDesc_B_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_B_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        AGridDesc_B_K0_M0_M1_K1 a_grid_desc_kbatch_k0_m0_m1_k1_;
        BGridDesc_B_K0_N0_N1_K1 b_grid_desc_kbatch_k0_n0_n1_k1_;
        CGridDesc_M0_M10_M11_N0_N10_N11 c_grid_desc_m0_m10_m11_n0_n10_n11_;

        // DefaultBlock2CTileMap block_2_ctile_map_;
        Block2CTileMap block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<> compute_ptr_offset_of_batch_;

        // element-wise op
        OutElementwiseOperation a_element_op_;
        WeiElementwiseOperation b_element_op_;
        InElementwiseOperation c_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_K_;
        const index_t Conv_C_;

        std::array<ck::index_t, NDimSpatial + 3> filter_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations_;
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

            std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {

            ShowInfo(arg);

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm GridwiseGemmDl_bkm_bkn_mn_v1r3 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Conv_G_;

            auto launch_kernel = [&](auto has_main_k_block_loop,
                                     auto has_double_tail_k_block_loop) {
                constexpr bool has_main_loop   = has_main_k_block_loop.value;
                constexpr bool has_double_loop = has_double_tail_k_block_loop.value;

                const auto kernel = kernel_batched_gemm_dlops_bwd_weight<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_B_K0_M0_M1_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_B_K0_N0_N1_K1>,
                    remove_reference_t<DeviceOp::CGridDesc_M0_M10_M11_N0_N10_N11>,
                    remove_reference_t<DeviceOp::Block2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<>,
                    has_main_loop,
                    has_double_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_c_grid_,
                                              arg.Conv_G_,
                                              arg.a_grid_desc_kbatch_k0_m0_m1_k1_,
                                              arg.b_grid_desc_kbatch_k0_n0_n1_k1_,
                                              arg.c_grid_desc_m0_m10_m11_n0_n10_n11_,
                                              arg.block_2_ctile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            const auto K0                    = arg.a_grid_desc_kbatch_k0_m0_m1_k1_.GetLength(I1);
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, true>{});
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, false>{});
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, false>{});
            }
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

        // DL version only supports split_k equal to 1
        if(arg.k_batch_ != 1)
            return false;

        if constexpr(!((NDimSpatial == 1 &&
                        (is_NWGC_GKXC_NWGK<InLayout, WeiLayout, OutLayout>() ||
                         is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>())) ||
                       (NDimSpatial == 2 &&
                        (is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                         is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>())) ||
                       (NDimSpatial == 3 &&
                        (is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                         is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))))
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_lengths_[spatial_offset + i] == 1 &&
                     arg.conv_filter_strides_[i] == 1 && arg.input_left_pads_[i] == 0 &&
                     arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // matrix A
        {
            auto srcVectorLengths = ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1{};
            if(srcVectorLengths[I2] != 1 || srcVectorLengths[I3] != 1)
            {
                return false;
            }
            if(K1 % srcVectorLengths[I4] != 0 || K0PerBlock % srcVectorLengths[I1] != 0)
            {
                return false;
            }

            const index_t K = arg.Conv_K_;

            if(K % (srcVectorLengths[I1] * srcVectorLengths[I4]) != 0)
            {
                return false;
            }
        }

        // matrix B
        {
            auto srcLoadLenghts   = BBlockTransferThreadSliceLengths_K0_N0_N1_K1{};
            auto srcVectorLengths = BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1{};
            if(srcVectorLengths[I1] != 1 || srcVectorLengths[I4] != 1)
            {
                return false;
            }
            if(srcLoadLenghts[I2] % srcVectorLengths[I2] != 0 ||
               srcLoadLenghts[I3] % srcVectorLengths[I3] != 0)
            {
                return false;
            }

            const index_t C = arg.Conv_K_;

            if(C % (srcVectorLengths[I2] * srcVectorLengths[I3]) != 0)
            {
                return false;
            }
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CThreadTransferDstScalarPerVector == 0))
        {
            std::cout << "Not surpport,because: arg.Conv_C_ % CThreadTransferDstScalarPerVector = "
                      << arg.Conv_C_ % CThreadTransferDstScalarPerVector << std::endl;
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(
            arg.a_grid_desc_kbatch_k0_m_k1_, arg.b_grid_desc_kbatch_k0_n_k1_, arg.c_grid_desc_m_n_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        a_g_n_c_wis_lengths, // input
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths, // weight
                        b_g_k_c_xs_strides,
                        e_g_n_k_wos_lengths, // output
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
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
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          a_g_n_c_wis_lengths, // input
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths, // weight
                                          b_g_k_c_xs_strides,
                                          e_g_n_k_wos_lengths, // output
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
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
        str << "DeviceGroupedConvBwdWeight_Dl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1
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
                      "instance_traits_device_grouped_conv_bwd_weight_dl.hpp "
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
