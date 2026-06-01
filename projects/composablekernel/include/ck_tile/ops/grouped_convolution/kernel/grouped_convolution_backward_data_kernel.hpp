// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_data.hpp"
#endif

namespace ck_tile {

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType_, typename TilePartitioner_>
struct GroupedConvBwdDataKernelArgs
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;

    using ConvToGemmTransformer =
        TransformConvBwdDataToGemm<GroupedConvTraitsType_::NDimSpatial,
                                   GroupedConvTraitsType_::ConvSpecialization,
                                   GroupedConvTraitsType_::VectorSizeA,
                                   GroupedConvTraitsType_::VectorSizeB,
                                   GroupedConvTraitsType_::VectorSizeC,
                                   true>; // Split N enabled
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdDataKernelArgs(const GroupedConvBwdDataHostArgs& args)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        const index_t X               = wei_g_k_c_xs_lengths[3];
        const index_t ConvStrideW     = conv_filter_strides[0];
        const index_t ConvDilationW   = conv_filter_dilations[0];
        const auto GcdStrideDilationW = gcd(ConvStrideW, ConvDilationW);
        const auto XTilde             = ConvStrideW / GcdStrideDilationW;

        for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
        {
            const auto XDotSlice = integer_divide_ceil(X - i_xtilde, XTilde);

            if(XDotSlice <= 0)
            {
                continue;
            }

            if(gemm_count >= MaxGroupedGemmGroupsNum)
            {
                gemm_count++;
                // Avoid array segfault
                continue;
            }

            tildes = {i_xtilde};

            ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                           wei_g_k_c_xs_lengths,
                                                           out_g_n_k_wos_lengths,
                                                           conv_filter_strides,
                                                           conv_filter_dilations,
                                                           input_left_pads,
                                                           input_right_pads,
                                                           tildes};

            auto grid_descs =
                conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                    GroupedConvTraitsType_::NDimSpatial>(1);

            a_grid_descs_m_k[gemm_count] = grid_descs.at(number<0>{});
            b_grid_descs_n_k[gemm_count] = grid_descs.at(number<1>{});
            c_grid_descs_m_n[gemm_count] = grid_descs.at(number<2>{});

            const index_t grid_size_grp =
                TilePartitioner::GridSize(c_grid_descs_m_n[gemm_count].get_length(I0),
                                          c_grid_descs_m_n[gemm_count].get_length(I1));

            block_starts[gemm_count] = grid_size_;
            block_ends[gemm_count]   = grid_size_ + grid_size_grp;

            grid_size_ += grid_size_grp;

            // Get the actual split N from transformer
            n_per_split = conv_to_gemm_transformer.GetN();
            original_n  = conv_to_gemm_transformer.GetOriginalN();
            n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

            ++gemm_count;
        }
        group_stride_a = args.K_; // A: Out NWGK
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>()); // B: Wei GKXC
        group_stride_c = args.C_;                                     // C: In  NWGC

        input_batch_stride  = args.C_ * args.G_ * args.input_spatial_lengths_[0];
        output_batch_stride = args.K_ * args.G_ * args.output_spatial_lengths_[0];

        GemmBatch = args.G_;
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdDataKernelArgs(const GroupedConvBwdDataHostArgs& args)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        const index_t Y               = wei_g_k_c_xs_lengths[3];
        const index_t X               = wei_g_k_c_xs_lengths[4];
        const index_t ConvStrideH     = conv_filter_strides[0];
        const index_t ConvStrideW     = conv_filter_strides[1];
        const index_t ConvDilationH   = conv_filter_dilations[0];
        const index_t ConvDilationW   = conv_filter_dilations[1];
        const auto GcdStrideDilationH = gcd(ConvStrideH, ConvDilationH);
        const auto GcdStrideDilationW = gcd(ConvStrideW, ConvDilationW);
        const auto YTilde             = ConvStrideH / GcdStrideDilationH;
        const auto XTilde             = ConvStrideW / GcdStrideDilationW;

        for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
        {
            for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
            {
                const auto YDotSlice = integer_divide_ceil(Y - i_ytilde, YTilde);
                const auto XDotSlice = integer_divide_ceil(X - i_xtilde, XTilde);

                if(XDotSlice * YDotSlice <= 0)
                {
                    continue;
                }

                if(gemm_count >= MaxGroupedGemmGroupsNum)
                {
                    gemm_count++;
                    // Avoid array segfault
                    continue;
                }

                tildes = {i_ytilde, i_xtilde};

                ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                               wei_g_k_c_xs_lengths,
                                                               out_g_n_k_wos_lengths,
                                                               conv_filter_strides,
                                                               conv_filter_dilations,
                                                               input_left_pads,
                                                               input_right_pads,
                                                               tildes};

                auto grid_descs = conv_to_gemm_transformer
                                      .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                                          GroupedConvTraitsType_::NDimSpatial>(1);

                a_grid_descs_m_k[gemm_count] = grid_descs.at(number<0>{});
                b_grid_descs_n_k[gemm_count] = grid_descs.at(number<1>{});
                c_grid_descs_m_n[gemm_count] = grid_descs.at(number<2>{});

                const index_t grid_size_grp =
                    TilePartitioner::GridSize(c_grid_descs_m_n[gemm_count].get_length(I0),
                                              c_grid_descs_m_n[gemm_count].get_length(I1));

                block_starts[gemm_count] = grid_size_;
                block_ends[gemm_count]   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // Get the actual split N from transformer
                n_per_split = conv_to_gemm_transformer.GetN();
                original_n  = conv_to_gemm_transformer.GetOriginalN();
                n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

                ++gemm_count;
            }
        }
        group_stride_a = args.K_; // A: Out NWGK
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>()); // B: Wei GKXC
        group_stride_c = args.C_;                                     // C: In  NWGC

        input_batch_stride =
            args.C_ * args.G_ * args.input_spatial_lengths_[0] * args.input_spatial_lengths_[1];
        output_batch_stride =
            args.K_ * args.G_ * args.output_spatial_lengths_[0] * args.output_spatial_lengths_[1];

        GemmBatch = args.G_;
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NDHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKZYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdDataKernelArgs(const GroupedConvBwdDataHostArgs& args)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1]),
                                 static_cast<index_t>(args.input_spatial_lengths_[2])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[2])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1]),
                                 static_cast<index_t>(args.output_spatial_lengths_[2])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1]),
                                 static_cast<index_t>(args.conv_filter_strides_[2])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1]),
                                 static_cast<index_t>(args.conv_filter_dilations_[2])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1]),
                                 static_cast<index_t>(args.input_left_pads_[2])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1]),
                                 static_cast<index_t>(args.input_right_pads_[2])};

        k_batch = args.k_batch;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        const index_t Z               = wei_g_k_c_xs_lengths[3];
        const index_t Y               = wei_g_k_c_xs_lengths[4];
        const index_t X               = wei_g_k_c_xs_lengths[5];
        const index_t ConvStrideD     = conv_filter_strides[0];
        const index_t ConvStrideH     = conv_filter_strides[1];
        const index_t ConvStrideW     = conv_filter_strides[2];
        const index_t ConvDilationD   = conv_filter_dilations[0];
        const index_t ConvDilationH   = conv_filter_dilations[1];
        const index_t ConvDilationW   = conv_filter_dilations[2];
        const auto GcdStrideDilationD = gcd(ConvStrideD, ConvDilationD);
        const auto GcdStrideDilationH = gcd(ConvStrideH, ConvDilationH);
        const auto GcdStrideDilationW = gcd(ConvStrideW, ConvDilationW);
        const auto ZTilde             = ConvStrideD / GcdStrideDilationD;
        const auto YTilde             = ConvStrideH / GcdStrideDilationH;
        const auto XTilde             = ConvStrideW / GcdStrideDilationW;

        for(index_t i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
        {
            for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
            {
                for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                {
                    const auto ZDotSlice = integer_divide_ceil(Z - i_ztilde, ZTilde);
                    const auto YDotSlice = integer_divide_ceil(Y - i_ytilde, YTilde);
                    const auto XDotSlice = integer_divide_ceil(X - i_xtilde, XTilde);

                    if(ZDotSlice * XDotSlice * YDotSlice <= 0)
                    {
                        continue;
                    }

                    if(gemm_count >= MaxGroupedGemmGroupsNum)
                    {
                        gemm_count++;
                        // Avoid array segfault
                        continue;
                    }

                    tildes = {i_ztilde, i_ytilde, i_xtilde};

                    ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                                   wei_g_k_c_xs_lengths,
                                                                   out_g_n_k_wos_lengths,
                                                                   conv_filter_strides,
                                                                   conv_filter_dilations,
                                                                   input_left_pads,
                                                                   input_right_pads,
                                                                   tildes};

                    auto grid_descs = conv_to_gemm_transformer
                                          .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                                              GroupedConvTraitsType_::NDimSpatial>(1);

                    a_grid_descs_m_k[gemm_count] = grid_descs.at(number<0>{});
                    b_grid_descs_n_k[gemm_count] = grid_descs.at(number<1>{});
                    c_grid_descs_m_n[gemm_count] = grid_descs.at(number<2>{});

                    const index_t grid_size_grp =
                        TilePartitioner::GridSize(c_grid_descs_m_n[gemm_count].get_length(I0),
                                                  c_grid_descs_m_n[gemm_count].get_length(I1));

                    block_starts[gemm_count] = grid_size_;
                    block_ends[gemm_count]   = grid_size_ + grid_size_grp;

                    grid_size_ += grid_size_grp;

                    // Get the actual split N from transformer
                    n_per_split = conv_to_gemm_transformer.GetN();
                    original_n  = conv_to_gemm_transformer.GetOriginalN();
                    n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

                    ++gemm_count;
                }
            }
        }

        group_stride_a = args.K_; // A: Out NWGK
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>()); // B: Wei GKXC
        group_stride_c = args.C_;                                     // C: In  NWGC

        input_batch_stride = args.C_ * args.G_ * args.input_spatial_lengths_[0] *
                             args.input_spatial_lengths_[1] * args.input_spatial_lengths_[2];
        output_batch_stride = args.K_ * args.G_ * args.output_spatial_lengths_[0] *
                              args.output_spatial_lengths_[1] * args.output_spatial_lengths_[2];

        GemmBatch = args.G_; // C: In  NWGC
    }

    static constexpr index_t MaxGroupedGemmGroupsNum = 128;

    using ABCGridDescs = remove_cvref_t<
        decltype(ConvToGemmTransformer{}.MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(1))>;

    using AGridDescMK = remove_cvref_t<decltype(ABCGridDescs{}[number<0>{}])>;
    using BGridDescNK = remove_cvref_t<decltype(ABCGridDescs{}[number<1>{}])>;
    using CGridDescMN = remove_cvref_t<decltype(ABCGridDescs{}[number<2>{}])>;

    static constexpr index_t NonSpatialDims = 3;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_right_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> tildes;

    index_t k_batch;
    index_t GemmBatch;
    index_t grid_size_ = 0;
    index_t gemm_count = 0;

    const void* out_ptr;
    void* in_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    const void* wei_ptr;

    array<AGridDescMK, MaxGroupedGemmGroupsNum> a_grid_descs_m_k;
    array<BGridDescNK, MaxGroupedGemmGroupsNum> b_grid_descs_n_k;
    array<CGridDescMN, MaxGroupedGemmGroupsNum> c_grid_descs_m_n;

    array<index_t, MaxGroupedGemmGroupsNum> block_starts;
    array<index_t, MaxGroupedGemmGroupsNum> block_ends;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;

    // Split-N support fields - initialize to safe defaults
    index_t n_splits            = 1; // Number of batch splits (e.g., 2 for 128→64×2)
    index_t n_per_split         = 1; // Batches per split (N_ from transformer)
    index_t original_n          = 1; // Original batch size before splitting
    index_t input_batch_stride  = 0; // Stride to next batch in input tensor
    index_t output_batch_stride = 0; // Stride to next batch in output tensor
};

/// @brief The Grouped Convolution Backward Data kernel template.
///
/// @paragraph Overview Overview
///            This class provides the grouped convolution backward data kernel template. By
///            semantic division of Implicit GEMM algorithm into following parts we achieve
///            flexible, versatile and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam GroupedConvTraitsType_       The type of class providing traits for grouped convolution.
/// @tparam TilePartitioner_            The type of class providing mapping of workgroup index into
/// the
///                                     output data tile to be calculated. It determines the
///                                     workgroup to data relationship (or in other words - which
///                                     data would be processed and calculated by which workgroup).
/// @tparam GemmPipeline_               The type of class which provides the core part of matrix
///                                     multiplication. This class should provide implementation of
///                                     data loading from global memory and performing block-wise
///                                     matrix multiplication. You can think of it as a work done by
///                                     single workgroup point of view.
/// @tparam EpiloguePipeline_           The type of class providing the final part of matrix
///                                     multiplication implementation. It is responsible for storing
///                                     results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                                     the output C tensor in global memory.
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionBackwardDataKernel
{
    static constexpr index_t NDimSpatial = GroupedConvTraitsType_::NDimSpatial;
    static constexpr ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using GemmALayout      = remove_cvref_t<typename GemmPipeline::ALayout>;
    using GemmBLayout      = remove_cvref_t<typename GemmPipeline::BLayout>;
    using GemmCLayout      = remove_cvref_t<typename GemmPipeline::CLayout>;

    using InLayout  = remove_cvref_t<typename GroupedConvTraitsType_::InLayout>;
    using WeiLayout = remove_cvref_t<typename GroupedConvTraitsType_::WeiLayout>;
    using OutLayout = remove_cvref_t<typename GroupedConvTraitsType_::OutLayout>;
    using DsLayout  = remove_cvref_t<typename GroupedConvTraitsType_::DsLayout>;

    using GemmDsLayout                  = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;

    using InDataType  = remove_cvref_t<typename GemmPipeline::ADataType>;
    using WeiDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;

    using OutDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using GroupedConvBwdDataKernelArgsSpecialized =
        GroupedConvBwdDataKernelArgs<GroupedConvTraitsType_, TilePartitioner>;
    static constexpr index_t MaxGroupedGemmGroupsNum =
        GroupedConvBwdDataKernelArgsSpecialized::MaxGroupedGemmGroupsNum;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                  "Not supported C GEMM layout!");
    static_assert(GroupedConvTraitsType_::ExplicitGemm == false, "Not supported yet!");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        static constexpr bool EnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
        constexpr auto NumGroupsToMerge        = GroupedConvTraitsType_::NumGroupsToMerge;
        // clang-format off
        return concat('_', "grouped_convolution_backward_data", 
            gemm_prec_str<InDataType, WeiDataType>(), 
            InLayout::name,
            WeiLayout::name,
            OutLayout::name,
            "gemm",
            GemmPipeline::GetName(),
            "epilogue",
            EpiloguePipeline::GetName(),
            getConvSpecializationString(ConvSpecialization),
            "MergedGroups",
            NumGroupsToMerge,
            "SplitImage",
            EnableSplitImage,
            "ExplicitGemm",
            GroupedConvTraitsType_::ExplicitGemm
        );
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetTypeString() { return GetName(); }

#ifdef CK_EXPERIMENTAL_BUILDER
    CK_TILE_HOST std::string GetInstanceString() const
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<GroupedConvolutionBackwardDataKernel>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_tile_grouped_convolution_backward_data.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<GroupedConvolutionBackwardDataKernel>();
    }
#endif

    CK_TILE_HOST static auto GridSize(const GroupedConvBwdDataKernelArgsSpecialized& kargs)
    {
        // enable batched grouped gemm
        return dim3(kargs.grid_size_, kargs.GemmBatch, kargs.n_splits * kargs.k_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr GroupedConvBwdDataKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvBwdDataHostArgs& hostArgs)
    {
        return GroupedConvBwdDataKernelArgsSpecialized(hostArgs);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE static auto
    MakeABlockWindow(const OutDataType* a_ptr,
                     const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                     const index_t group_id,
                     const index_t i_m,
                     const index_t i_k)
    {
        // Step 1: Create tensor view for A (Out tensor)
        const auto& a_tensor_view =
            make_tensor_view<address_space_enum::global>(a_ptr, kargs.a_grid_descs_m_k[group_id]);

        // Step 2: Create padded view
        const auto& a_pad_view = pad_tensor_view(
            a_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            sequence<true, true>{});

        // Step 3: Create tile window
        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {i_m, i_k});

        return a_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeBBlockWindow(const InDataType* b_ptr,
                     const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                     const index_t group_id,
                     const index_t i_n,
                     const index_t i_k)
    {
        // Step 1: Create tensor view for B (Weight tensor)
        const auto& b_tensor_view =
            make_tensor_view<address_space_enum::global>(b_ptr, kargs.b_grid_descs_n_k[group_id]);

        // Step 2: Create padded view
        const auto& b_pad_view = pad_tensor_view(
            b_tensor_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<true, true>{});

        // Step 3: Create tile window
        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_k, i_n});

        return b_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                      const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                      const index_t group_id,
                      const index_t i_m,
                      const index_t i_n)
    {
        // Create D tensor block windows
        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                // Step 1: Create tensor view for D
                const auto& d_tensor_view = make_tensor_view<address_space_enum::global>(
                    static_cast<const OutDataType*>(ds_ptr[i]), kargs.c_grid_descs_m_n[group_id]);

                // Step 2: Create padded view
                const auto& d_pad_view =
                    pad_tensor_view(d_tensor_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::NPerBlock>{}),
                                    sequence<true, true>{});

                // Step 3: Create tile window
                return make_tile_window(d_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            },
            number<NumDTensor>{});

        return ds_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeCBlockWindow(WeiDataType* c_ptr,
                     const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                     const index_t group_id,
                     const index_t i_m,
                     const index_t i_n)
    {
        // Step 1: Create tensor view for C (Input tensor)
        const auto& c_tensor_view = make_tensor_view<address_space_enum::global, DstInMemOp>(
            c_ptr, kargs.c_grid_descs_m_n[group_id]);

        // Step 2: Create padded view
        const auto& c_pad_view = pad_tensor_view(
            c_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<true, true>{});

        // Step 3: Create tile window
        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return c_block_window;
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const GroupedConvBwdDataKernelArgsSpecialized& kargs)
    {
        if constexpr(GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                     is_any_of<OutDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        if(kargs.gemm_count > MaxGroupedGemmGroupsNum)
        {
            return false;
        }

        const index_t ConvK = kargs.wei_g_k_c_xs_lengths[number<1>{}];
        const index_t ConvC = kargs.wei_g_k_c_xs_lengths[number<2>{}];

        // check ConvSpecialization
        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                const index_t ConvStride = kargs.conv_filter_strides[i];
                const index_t LeftPad    = kargs.input_left_pads[i];
                const index_t RightPad   = kargs.input_right_pads[i];

                if(!(SpatialDim == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                const index_t LeftPad    = kargs.input_left_pads[i];
                const index_t RightPad   = kargs.input_right_pads[i];

                if(!(SpatialDim == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if(ConvC != 1)
            {
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = kargs.wei_g_k_c_xs_lengths[i + I3];

                if(filter_spatial_dim != I3)
                {
                    return false;
                }
            }
        }

        namespace ctc = tensor_layout::convolution;

        if constexpr(std::is_same_v<InLayout, ctc::NWGC> || std::is_same_v<InLayout, ctc::NHWGC> ||
                     std::is_same_v<InLayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(ConvC % GroupedConvTraitsType_::VectorSizeB != 0)
            {
                CK_TILE_ERROR("Conv C is not a multiple of vector load size for input image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported input layout!");
            return false;
        }

        // FIXME: layout
        if constexpr(std::is_same_v<WeiLayout, ctc::GKXC> ||
                     std::is_same_v<WeiLayout, ctc::GKYXC> ||
                     std::is_same_v<WeiLayout, ctc::GKZYXC>)
        {
            if(ConvC % GroupedConvTraitsType_::VectorSizeC != 0)
            {
                CK_TILE_ERROR("Conv C is not a multiple of vector load size for weight!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported weight layout!");
            return false;
        }

        if constexpr(std::is_same_v<OutLayout, ctc::NWGK> ||
                     std::is_same_v<OutLayout, ctc::NHWGK> ||
                     std::is_same_v<OutLayout, ctc::NDHWGK>)
        {
            if(ConvK % GroupedConvTraitsType_::VectorSizeA != 0)
            {
                CK_TILE_ERROR("Conv K is not a multiple of vector store size for output image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported output layout!");
            return false;
        }

        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const OutDataType* a_ptr,
                        const InDataType* b_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        WeiDataType* c_ptr,
                        const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                        const index_t group_id)
    {
        static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");
        static_assert(!GemmPipeline::BlockGemmShape::PermuteB, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(
                a_ptr,
                kargs.a_grid_descs_m_k[group_id]); // A: out
        }();

        const auto& b_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(
                b_ptr,
                kargs.b_grid_descs_n_k[group_id]); // B: weight
        }();

        const auto& c_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global, DstInMemOp>(
                c_ptr, kargs.c_grid_descs_m_n[group_id]);
        }();

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                static_assert(std::is_same_v<std::tuple_element_t<i, DsLayout>, OutLayout>,
                              "Not supported!");
                static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                              "Not supported!");
                static_assert(std::is_same_v<std::tuple_element_t<i, DsDataType>, OutDataType>,
                              "Not supported!");

                return make_tensor_view<address_space_enum::global>(
                    static_cast<OutDataType*>(ds_ptr[i]), kargs.c_grid_descs_m_n[group_id]);
            },
            number<NumDTensor>{});

        return make_tuple(a_tensor_view, b_tensor_view, ds_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            return pad_tensor_view(a_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                              number<TilePartitioner::KPerBlock>{}),
                                   sequence<true, true>{});
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            return pad_tensor_view(b_tensor_view,
                                   make_tuple(number<TilePartitioner::KPerBlock>{},
                                              number<TilePartitioner::NPerBlock>{}),
                                   sequence<true, true>{});
        }();

        const auto& ds_tensor_view = views.at(I2);
        const auto& ds_pad_view    = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(ds_tensor_view[i],
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<true, true>{});
            },
            number<NumDTensor>{});

        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I3);
            return pad_tensor_view(c_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                              number<TilePartitioner::NPerBlock>{}),
                                   sequence<true, true>{});
        }();

        return make_tuple(a_pad_view, b_pad_view, ds_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto MakeGemmTileWindows(const PadView& views,
                                                   const index_t i_m,
                                                   const index_t i_n,
                                                   const index_t i_k)
    {
        const auto& a_pad_view  = views.at(I0);
        const auto& b_pad_view  = views.at(I1);
        const auto& ds_pad_view = views.at(I2);
        const auto& c_pad_view  = views.at(I3);

        const auto& a_block_window = [&]() {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {i_m, i_k});
        }();

        const auto& b_block_window = [&]() {
            return make_tile_window(b_pad_view,
                                    make_tuple(number<TilePartitioner::KPerBlock>{},
                                               number<TilePartitioner::NPerBlock>{}),
                                    {i_k, i_n});
        }();

        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(ds_pad_view[i],
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            },
            number<NumDTensor>{});

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(a_block_window, b_block_window, ds_block_window, c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs Grouped Convolution Backward Data kernel arguments
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const OutDataType* a_ptr,
                                       const InDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       WeiDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                                       const index_t splitted_k,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const index_t block_idx_k,
                                       const index_t group_id)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            MakeABlockWindow(a_ptr, kargs, group_id, block_idx_m, block_idx_k);
        const auto& b_block_window =
            MakeBBlockWindow(b_ptr, kargs, group_id, block_idx_n, block_idx_k);
        const auto& d_block_window =
            MakeDBlockWindows(ds_ptr, kargs, group_id, block_idx_m, block_idx_n);

        const index_t num_loop  = amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitted_k));
        const bool has_hot_loop = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, has_hot_loop, tail_num, smem_ptr_0);

        const index_t k_batch = amd_wave_read_first_lane(kargs.k_batch);

        // Run Epilogue Pipeline with k_batch dispatch
        if(k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, group_id, block_idx_m, block_idx_n);

            EpiloguePipeline{}
                .template operator()<decltype(c_block_window), decltype(c_block_tile)>(
                    c_block_window, c_block_tile, d_block_window, smem_ptr_0);
        }
        else
        {
            if constexpr(!(GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value))
            {
                auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                    c_ptr, kargs, group_id, block_idx_m, block_idx_n);

                EpiloguePipeline{}
                    .template operator()<decltype(c_block_window), decltype(c_block_tile)>(
                        c_block_window, c_block_tile, d_block_window, smem_ptr_0);
            }
        }
    }

    CK_TILE_DEVICE index_t FindGroupId(const GroupedConvBwdDataKernelArgsSpecialized& kargs,
                                       index_t block_id) const
    {
        index_t left     = 0;
        index_t right    = kargs.gemm_count;
        index_t group_id = index_t((left + right) >> 1);

        while((!(block_id >= kargs.block_starts[group_id] &&
                 block_id < kargs.block_ends[group_id])) &&
              left <= right)
        {
            if(block_id < kargs.block_starts[group_id])
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) >> 1);
        }

        return group_id;
    }

    CK_TILE_DEVICE void operator()(GroupedConvBwdDataKernelArgsSpecialized& kargs) const
    {
        const auto blockIdX    = amd_wave_read_first_lane(blockIdx.x);
        const index_t group_id = FindGroupId(kargs, blockIdX);

        const auto [iM, iN] = OffsettedTile1DPartitioner<TilePartitioner>::GetOffsetedTileIndex(
            kargs.block_starts[group_id],
            kargs.c_grid_descs_m_n[group_id].get_length(I0),
            kargs.c_grid_descs_m_n[group_id].get_length(I1));

        const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const auto blockIdY       = amd_wave_read_first_lane(blockIdx.y);
        const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * blockIdY);
        const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * blockIdY);
        const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * blockIdY);

        const auto blockIdZ = amd_wave_read_first_lane(blockIdx.z);

        // SplitN
        const index_t split_n_idx = __builtin_amdgcn_readfirstlane(blockIdZ / kargs.k_batch);
        const index_t split_n_offset =
            __builtin_amdgcn_readfirstlane(split_n_idx * kargs.n_per_split);

        const long_index_t output_batch_offset =
            static_cast<long_index_t>(split_n_offset) *
            static_cast<long_index_t>(kargs.output_batch_stride);
        const long_index_t input_batch_offset = static_cast<long_index_t>(split_n_offset) *
                                                static_cast<long_index_t>(kargs.input_batch_stride);

        // SplitK
        const index_t split_k_idx =
            __builtin_amdgcn_readfirstlane(blockIdZ - split_n_idx * kargs.k_batch);

        const index_t gemm_k = kargs.a_grid_descs_m_k[group_id].get_length(I1);

        constexpr auto K1   = TilePartitioner::KPerBlock;
        const index_t K_t   = amd_wave_read_first_lane(kargs.k_batch * K1);
        const index_t KRead = amd_wave_read_first_lane((gemm_k + K_t - 1) / K_t * K1);

        const index_t i_k        = amd_wave_read_first_lane(split_k_idx * KRead);
        const index_t splitted_k = amd_wave_read_first_lane(KRead);

        // options
        // conv_bwd_data = Out * Weight = In
        const OutDataType* a_ptr =
            static_cast<const OutDataType*>(kargs.out_ptr) + group_offset_a + output_batch_offset;
        const WeiDataType* b_ptr = static_cast<const WeiDataType*>(kargs.wei_ptr) + group_offset_b;
        InDataType* c_ptr =
            static_cast<InDataType*>(kargs.in_ptr) + group_offset_c + input_batch_offset;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];
        RunGemm(a_ptr,
                b_ptr,
                kargs.ds_ptr,
                c_ptr,
                smem_ptr,
                kargs,
                splitted_k,
                i_m,
                i_n,
                i_k,
                group_id);
    }
};

} // namespace ck_tile
