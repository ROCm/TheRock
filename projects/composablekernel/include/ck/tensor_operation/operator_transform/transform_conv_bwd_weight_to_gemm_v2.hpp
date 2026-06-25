// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"

namespace ck {
namespace tensor_operation {

/**
 * @brief Transform conv bwd weight to gemm v2
 *
 * This version does following things:
 * 1. Merge KBatch with K0 to align descriptor with universal gemm
 * 2. Merge Batch with M and N dimension. It allows to increase compute in
 *    case of small M and N. It also allows to vector load and store in case of
 *    K = 1, C = 1 and NHWGC layout.
 */
template <index_t NDimSpatial,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t GemmK1Number,
          index_t K0PerBlock,
          index_t NumGroupsToMerge,
          device::ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization>
struct TransformConvBwdWeightToGemmV2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    template <index_t NDim, typename enable_if<NDim == 1, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const index_t N,
                       const index_t Wo,
                       const index_t K,
                       const std::array<index_t, NDimSpatial + 3>& output_strides)
    {
        const index_t BatchStride = output_strides[0];
        const index_t WoStride    = output_strides[3];
        const auto KStride        = Number<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Wo, NumGroupsToMerge, K),
                                            make_tuple(WoStride, BatchStride, KStride));
    }

    template <index_t NDim, typename enable_if<NDim == 1, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const index_t N,
                      const index_t Wi,
                      const index_t C,
                      const std::array<index_t, NDimSpatial + 3>& input_strides)
    {
        const index_t BatchStride = input_strides[0];
        const index_t NStride     = input_strides[1];
        const index_t WiStride    = input_strides[3];
        const auto CStride        = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Wi, NumGroupsToMerge, C),
                                                make_tuple(WiStride, BatchStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Wi, NumGroupsToMerge, C),
                make_tuple(NStride, WiStride, BatchStride, CStride));
        }
    }

    template <index_t NDim, typename enable_if<NDim == 1, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const index_t K,
                       const index_t X,
                       const index_t C,
                       const std::array<index_t, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride     = Number<1>{};
        const auto KStride     = weights_strides[1];
        const auto XStride     = weights_strides[3];
        const auto BatchStride = weights_strides[0];
        // Add NumGroupsToMerge for Batch+M dimension and, 1 as a placehorder
        // for Batch+N dimension
        const auto desc = make_naive_tensor_descriptor(
            make_tuple(NumGroupsToMerge, K, X, 1, C),
            make_tuple(BatchStride, KStride, XStride, BatchStride, CStride));
        // Padd 1 to NumGroupsToMerge
        const auto padded_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(NumGroupsToMerge),
                       make_pass_through_transform(K),
                       make_pass_through_transform(X),
                       make_pad_transform(1, 0, NumGroupsToMerge - 1),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));
        // We need only matrices from diagonal. Xor returns 0 for the same
        // values. So if matrices is not on diagonal then it will be stored in padding.
        // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
        static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                      NumGroupsToMerge == 8 || NumGroupsToMerge == 16 || NumGroupsToMerge == 32 ||
                      NumGroupsToMerge == 64);
        const auto unmerged_padded_desc = transform_tensor_descriptor(
            padded_desc,
            make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                       make_pass_through_transform(K),
                       make_pass_through_transform(X),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}));
        // Merge To M, N
        return transform_tensor_descriptor(
            unmerged_padded_desc,
            make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K)),
                       make_merge_transform(make_tuple(X, NumGroupsToMerge, C))),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const index_t N,
                       const index_t Ho,
                       const index_t Wo,
                       const index_t K,
                       const std::array<index_t, NDimSpatial + 3>& output_strides)
    {
        const index_t BatchStride = output_strides[0];
        const index_t WoStride    = output_strides[4];
        const auto KStride        = Number<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, NumGroupsToMerge, K),
                                            make_tuple(WoStride, BatchStride, KStride));
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const index_t N,
                      const index_t Hi,
                      const index_t Wi,
                      const index_t C,
                      const std::array<index_t, NDimSpatial + 3>& input_strides)
    {
        const index_t BatchStride = input_strides[0];
        const index_t NStride     = input_strides[1];
        const index_t HiStride    = input_strides[3];
        const index_t WiStride    = input_strides[4];
        const auto CStride        = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Hi * Wi, NumGroupsToMerge, C),
                                                make_tuple(WiStride, BatchStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, NumGroupsToMerge, C),
                make_tuple(NStride, HiStride, WiStride, BatchStride, CStride));
        }
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const index_t K,
                       const index_t Y,
                       const index_t X,
                       const index_t C,
                       const std::array<index_t, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride     = Number<1>{};
        const auto KStride     = weights_strides[1];
        const auto XStride     = weights_strides[4];
        const auto BatchStride = weights_strides[0];
        // Add NumGroupsToMerge for Batch+M dimension and, 1 as a placehorder
        // for Batch+N dimension
        const auto desc = make_naive_tensor_descriptor(
            make_tuple(NumGroupsToMerge, K, Y * X, 1, C),
            make_tuple(BatchStride, KStride, XStride, BatchStride, CStride));
        // Padd 1 to NumGroupsToMerge
        const auto padded_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(NumGroupsToMerge),
                       make_pass_through_transform(K),
                       make_pass_through_transform(Y * X),
                       make_pad_transform(1, 0, NumGroupsToMerge - 1),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));
        // We need only matrices from diagonal. Xor returns 0 for the same
        // values. So if matrices is not on diagonal then it will be stored in padding.
        // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
        static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                      NumGroupsToMerge == 8 || NumGroupsToMerge == 16 || NumGroupsToMerge == 32 ||
                      NumGroupsToMerge == 64);
        const auto unmerged_padded_desc = transform_tensor_descriptor(
            padded_desc,
            make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                       make_pass_through_transform(K),
                       make_pass_through_transform(Y * X),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}));
        // Merge To M, N
        return transform_tensor_descriptor(
            unmerged_padded_desc,
            make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K)),
                       make_merge_transform(make_tuple(Y * X, NumGroupsToMerge, C))),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const index_t N,
                       const index_t Do,
                       const index_t Ho,
                       const index_t Wo,
                       const index_t K,
                       const std::array<index_t, NDimSpatial + 3>& output_strides)
    {
        const index_t BatchStride = output_strides[0];
        const index_t WoStride    = output_strides[5];
        const auto KStride        = Number<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Do * Ho * Wo, NumGroupsToMerge, K),
                                            make_tuple(WoStride, BatchStride, KStride));
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const index_t N,
                      const index_t Di,
                      const index_t Hi,
                      const index_t Wi,
                      const index_t C,
                      const std::array<index_t, NDimSpatial + 3>& input_strides)
    {
        const index_t BatchStride = input_strides[0];
        const index_t NStride     = input_strides[1];
        const index_t DiStride    = input_strides[3];
        const index_t HiStride    = input_strides[4];
        const index_t WiStride    = input_strides[5];
        const auto CStride        = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Di * Hi * Wi, NumGroupsToMerge, C),
                                                make_tuple(WiStride, BatchStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, NumGroupsToMerge, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, BatchStride, CStride));
        }
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const index_t K,
                       const index_t Z,
                       const index_t Y,
                       const index_t X,
                       const index_t C,
                       const std::array<index_t, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride     = Number<1>{};
        const auto KStride     = weights_strides[1];
        const auto XStride     = weights_strides[5];
        const auto BatchStride = weights_strides[0];
        // Add NumGroupsToMerge for Batch+M dimension and, 1 for placehord for Batch+N dimension
        const auto desc = make_naive_tensor_descriptor(
            make_tuple(NumGroupsToMerge, K, Z * Y * X, 1, C),
            make_tuple(BatchStride, KStride, XStride, BatchStride, CStride));
        // Padd 1 to NumGroupsToMerge
        const auto padded_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(NumGroupsToMerge),
                       make_pass_through_transform(K),
                       make_pass_through_transform(Z * Y * X),
                       make_pad_transform(1, 0, NumGroupsToMerge - 1),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));
        // We need only matrices from diagonal. Xor returns 0 for the same
        // values. So if matrices is not on diagonal then it will be stored in padding.
        // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
        static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                      NumGroupsToMerge == 8 || NumGroupsToMerge == 16 || NumGroupsToMerge == 32 ||
                      NumGroupsToMerge == 64);
        const auto unmerged_padded_desc = transform_tensor_descriptor(
            padded_desc,
            make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                       make_pass_through_transform(K),
                       make_pass_through_transform(Z * Y * X),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}));
        // Merge To M, N
        return transform_tensor_descriptor(
            unmerged_padded_desc,
            make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K)),
                       make_merge_transform(make_tuple(Z * Y * X, NumGroupsToMerge, C))),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <index_t NDim, typename enable_if<NDim == 1, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const index_t N,
        const index_t K,
        const index_t C,
        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
        const std::array<index_t, NDimSpatial + 3>& input_strides,
        const std::array<index_t, NDimSpatial + 3>& weights_strides,
        const std::array<index_t, NDimSpatial + 3>& output_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack   = false,
        const bool use_full_batch_kindex = false)
    {
        using namespace ck;

        const index_t Wi = input_spatial_lengths[0];

        const index_t Wo = output_spatial_lengths[0];

        const index_t X = filter_spatial_lengths[0];

        const index_t ConvStrideW = conv_filter_strides[0];

        const index_t ConvDilationW = conv_filter_dilations[0];

        const index_t InLeftPadW = input_left_pads[0];

        const index_t InRightPadW = input_right_pads[0];

        const index_t GemmKTotal = N * Wo;
        const index_t GemmM      = K * NumGroupsToMerge;
        const index_t GemmN      = C * X * NumGroupsToMerge;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        // When use_full_batch_kindex=true, create full-batch descriptors (V1 mode) for gridwise
        // kernel compatibility
        const index_t KBatchDim = (split_k_offset_hack && !use_full_batch_kindex) ? 1 : GemmKBatch;
        const index_t GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmN / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(NumGroupsToMerge),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(X, NumGroupsToMerge, C)),
                           make_merge_transform(make_tuple(N, Wo))),
                make_tuple(Sequence<1, 3, 4>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }

    } // function end

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const index_t N,
        const index_t K,
        const index_t C,
        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
        const std::array<index_t, NDimSpatial + 3>& input_strides,
        const std::array<index_t, NDimSpatial + 3>& weights_strides,
        const std::array<index_t, NDimSpatial + 3>& output_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack   = false,
        const bool use_full_batch_kindex = false)
    {
        using namespace ck;

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t GemmKTotal = N * Ho * Wo;
        const index_t GemmM      = K * NumGroupsToMerge;
        const index_t GemmN      = C * X * Y * NumGroupsToMerge;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        // When use_full_batch_kindex=true, create full-batch descriptors (V1 mode) for gridwise
        // kernel compatibility
        const index_t KBatchDim = (split_k_offset_hack && !use_full_batch_kindex) ? 1 : GemmKBatch;
        const index_t GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Ho, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Hi, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, Y, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmN / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(NumGroupsToMerge),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5>{},
                           Sequence<6>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Y, X, NumGroupsToMerge, C)),
                           make_merge_transform(make_tuple(N, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const index_t N,
        const index_t K,
        const index_t C,
        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
        const std::array<index_t, NDimSpatial + 3>& input_strides,
        const std::array<index_t, NDimSpatial + 3>& weights_strides,
        const std::array<index_t, NDimSpatial + 3>& output_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack   = false,
        const bool use_full_batch_kindex = false)
    {
        using namespace ck;

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t Z = filter_spatial_lengths[0];
        const index_t Y = filter_spatial_lengths[1];
        const index_t X = filter_spatial_lengths[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        const index_t ConvDilationD = conv_filter_dilations[0];
        const index_t ConvDilationH = conv_filter_dilations[1];
        const index_t ConvDilationW = conv_filter_dilations[2];

        const index_t InLeftPadD = input_left_pads[0];
        const index_t InLeftPadH = input_left_pads[1];
        const index_t InLeftPadW = input_left_pads[2];

        const index_t InRightPadD = input_right_pads[0];
        const index_t InRightPadH = input_right_pads[1];
        const index_t InRightPadW = input_right_pads[2];

        const index_t GemmKTotal = N * Do * Ho * Wo;
        const index_t GemmM      = K * NumGroupsToMerge;
        const index_t GemmN      = C * Z * X * Y * NumGroupsToMerge;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const index_t GemmKBatch = batch_k;
        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        // When use_full_batch_kindex=true, create full-batch descriptors (V1 mode) for gridwise
        // kernel compatibility
        const index_t KBatchDim = (split_k_offset_hack && !use_full_batch_kindex) ? 1 : GemmKBatch;
        const index_t GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Do, Ho, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Di, Hi, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, Z, Y, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmN / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(
                    make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                    make_merge_transform(make_tuple(NumGroupsToMerge, GemmM / NumGroupsToMerge))),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}));

            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_dip_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(NumGroupsToMerge),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{},
                           Sequence<8>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Z, Y, X, NumGroupsToMerge, C)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 7, 8>{}, Sequence<0, 2, 4, 6>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim * GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    } // function end
};

} // namespace tensor_operation
} // namespace ck
