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
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NDimSpatial,
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
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          index_t NumGemmKPrefetchStage                        = 1,
          LoopScheduler LoopSched                              = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer                      = ck::PipelineVersion::v1,
          typename ck::enable_if<NDimSpatial == 3, bool>::type = false>
struct DeviceGroupedConvBwdWeight_Wmma_CShuffle
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
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
    using DeviceOp = DeviceGroupedConvBwdWeight_Wmma_CShuffle;

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

    static constexpr auto GemmK1Number = Number<K1>{};
    static constexpr index_t KPerBlock = K0PerBlock * GemmK1Number;

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const index_t N,
                       const index_t Do,
                       const index_t Ho,
                       const index_t Wo,
                       const index_t K,
                       const std::array<index_t, NDimSpatial + 3>& output_strides)
    {
        const index_t WoStride = output_strides[5];
        const auto KStride     = Number<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Do * Ho * Wo, K),
                                            make_tuple(WoStride, KStride));
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const index_t N,
                      const index_t Di,
                      const index_t Hi,
                      const index_t Wi,
                      const index_t C,
                      const std::array<index_t, NDimSpatial + 3>& input_strides)
    {
        const index_t NStride  = input_strides[1];
        const index_t DiStride = input_strides[3];
        const index_t HiStride = input_strides[4];
        const index_t WiStride = input_strides[5];
        const auto CStride     = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Di * Hi * Wi, C),
                                                make_tuple(WiStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));
        }
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const index_t K,
                       const index_t Z,
                       const index_t Y,
                       const index_t X,
                       const index_t C,
                       const std::array<index_t, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride = Number<1>{};
        const auto KStride = weights_strides[1];
        return make_naive_tensor_descriptor(make_tuple(K, Z * Y * X * C),
                                            make_tuple(KStride, CStride));
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
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
        const std::array<index_t, NDimSpatial>& input_right_pads)
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
        const index_t GemmM      = K;
        const index_t GemmN      = C * Z * X * Y;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const index_t GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock) * K0PerBlock;
        const index_t GemmKPad = GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Do, Ho, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Di, Hi, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, Z, Y, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // B: input tensor
            const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
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

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // Pad
            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_pad_grid_desc =
                transform_tensor_descriptor(
                    out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                    make_tuple(make_pass_through_transform(GemmK0),
                               make_right_pad_transform(GemmM, PadGemmM),
                               make_pass_through_transform(GemmK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_pad_grid_desc =
                transform_tensor_descriptor(
                    in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                    make_tuple(make_pass_through_transform(GemmK0),
                               make_right_pad_transform(GemmN, PadGemmN),
                               make_pass_through_transform(GemmK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_pad_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_pad_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const index_t dim = 1;
        const std::array<index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<index_t, NDimSpatial> params{1, 1, 1};
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(dim,
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
                                                                  params);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using CShuffleDataType = AccDataType;

    using GridwiseGemm = GridwiseGemmMultipleD_Wmma<
        // DataType Family
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        Tuple<>,
        CDataType,
        // InMemory Data Descriptor
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        Tuple<>,
        CGridDesc_M_N,
        // ElementwiseOp Family
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // Tiling Family
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWMMA,
        NPerWMMA,
        K1,
        MRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        true,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        true,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        NumGemmKPrefetchStage,
        LoopSched,
        PipelineVer>;

    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(Tuple<>{}));

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}));

    using Block2CTileMap = decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(
        CGridDesc_M_N{}, I1 /* M01 */, I1 /* N01 */));

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
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_c_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_ctile_map_{},
              compute_ptr_offset_of_batch_{},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_G_{a_g_n_c_wis_lengths[0]},
              Conv_N_{a_g_n_c_wis_lengths[1]},
              Conv_K_{b_g_k_c_xs_lengths[1]},
              Conv_C_{a_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            constexpr index_t spatial_offset = 3;
            std::copy(begin(a_g_n_c_wis_lengths) + spatial_offset,
                      end(a_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(b_g_k_c_xs_lengths) + spatial_offset,
                      end(b_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(e_g_n_k_wos_lengths) + spatial_offset,
                      end(e_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

            k_batch_ = split_k;

            const auto descs =
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                    Conv_N_,
                    Conv_K_,
                    Conv_C_,
                    input_spatial_lengths_,
                    filter_spatial_lengths_,
                    output_spatial_lengths_,
                    a_g_n_c_wis_strides,
                    b_g_k_c_xs_strides,
                    e_g_n_k_wos_strides,
                    conv_filter_strides,
                    conv_filter_dilations,
                    input_left_pads,
                    input_right_pads);

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_            = descs[I2];

            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(
                c_grid_desc_m_n_, I1 /* M01 */, I1 /* N01 */);

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = e_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{});

            if(GridwiseGemm::CheckValidity(a_grid_desc_kbatch_k0_m_k1_,
                                           b_grid_desc_kbatch_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;

        Block2CTileMap block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<> compute_ptr_offset_of_batch_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<index_t, NDimSpatial> input_spatial_lengths_;
        std::array<index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<index_t, NDimSpatial>& input_left_pads_;
        const std::array<index_t, NDimSpatial>& input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void Print(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                Print(arg);
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                            arg.b_grid_desc_kbatch_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmMultipleD_k0mk1_k0nk1_mn_wmma_cshuffle has invalid "
                    "setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.Conv_G_;

            const auto K0 = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_grouped_conv_multiple_d_wmma_cshuffle<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    typename GridwiseGemm::DsGridPointer,
                    CDataType,
                    OutElementwiseOperation,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    AGridDesc_K0_M_K1,
                    BGridDesc_K0_N_K1,
                    DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    ComputePtrOffsetOfStridedBatch<>,
                    has_main_loop>;

                using EmptyTuple = Tuple<>;
                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              EmptyTuple{}, // Ds
                                              arg.p_c_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.c_element_op_,
                                              arg.Conv_G_,
                                              arg.a_grid_desc_kbatch_k0_m_k1_,
                                              arg.b_grid_desc_kbatch_k0_n_k1_,
                                              DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock{},
                                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_ctile_map_,
                                              arg.compute_ptr_offset_of_batch_);
            };

            if(has_main_k0_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
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
        // check device
        if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // TODO: Add support for split_k > 1
        if(arg.k_batch_ != 1)
        {
            return false;
        }

        if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                       is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's a 1x1 convolution with stride=1 and no padding
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
        if(!(arg.Conv_C_ % CShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
                                           arg.b_grid_desc_kbatch_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
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
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const index_t split_k)
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
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const index_t split_k) override
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
        str << "DeviceGroupedConvBwdWeight_Wmma_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle << ", "
            << CShuffleBlockTransferScalarPerVector_NPerBlock
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
                      "instance_traits_device_grouped_conv_bwd_weight_wmma_cshuffle.hpp "
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
