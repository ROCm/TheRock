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
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

#include "ck_tile/ops/grouped_convolution/utils/split_k_utils.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_weight.hpp"
#endif

namespace ck_tile {

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType_>
struct GroupedConvBwdWeightKernelArgs
{

    using ConvToGemmTransformer =
        TransformConvBwdWeightToGemm<GroupedConvTraitsType_::NDimSpatial,
                                     GroupedConvTraitsType_::ConvSpecialization,
                                     GroupedConvTraitsType_::VectorSizeA,
                                     GroupedConvTraitsType_::VectorSizeB,
                                     GroupedConvTraitsType_::VectorSizeC,
                                     GroupedConvTraitsType_::NumGroupsToMerge>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch
                      << ", NumGroupsPerBatch: " << NumGroupsPerBatch << ", k_batch: " << k_batch
                      << std::endl;
        }
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NHWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NHWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKYXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch
                      << ", NumGroupsPerBatch: " << NumGroupsPerBatch << ", k_batch: " << k_batch
                      << std::endl;
        }
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NDHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKZYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NDHWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NDHWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKZYXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "GemmM: " << GemmM << ", GemmN: " << GemmN << ", GemmK: " << GemmK
                      << ", GemmBatch: " << GemmBatch
                      << ", NumGroupsPerBatch: " << NumGroupsPerBatch << ", k_batch: " << k_batch
                      << std::endl;
        }
    }

    using ABCGridDescs = remove_cvref_t<
        decltype(ConvToGemmTransformer{}.MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N())>;

    using AGridDescKM = remove_cvref_t<decltype(ABCGridDescs{}[number<0>{}])>;
    using BGridDescKN = remove_cvref_t<decltype(ABCGridDescs{}[number<1>{}])>;
    using CGridDescMN = remove_cvref_t<decltype(ABCGridDescs{}[number<2>{}])>;

    static constexpr index_t NonSpatialDims = 3;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_right_pads;

    index_t k_batch;
    index_t GemmM;
    index_t GemmN;
    index_t GemmK;
    index_t GemmBatch;
    index_t NumGroupsPerBatch;

    const void* out_ptr;
    const void* in_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* wei_ptr;

    AGridDescKM a_grid_desc_k_m;
    BGridDescKN b_grid_desc_k_n;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;
};

/// @brief The Grouped Convolution Backward Weight kernel template.
///
/// @paragraph Overview Overview
///            This class provides the grouped convolution backward weight kernel template. By
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
/// @tparam GroupedConvTraitsType_      The type of class providing traits for grouped convolution.
/// @tparam TilePartitioner_            The type of class providing mapping of workgroup index into
///                                     the output data tile to be calculated. It determines the
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
struct GroupedConvolutionBackwardWeightKernel
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

    using OutDataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using InDataType  = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    using WeiDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using GroupedConvBwdWeightKernelArgsSpecialized =
        GroupedConvBwdWeightKernelArgs<GroupedConvTraitsType_>;

    static constexpr bool IsSplitKSupported = true;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::ColumnMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(GroupedConvTraitsType_::ExplicitGemm == false ||
                      GroupedConvTraitsType_::NumGroupsToMerge == 1,
                  "Not supported!");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        static constexpr bool EnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
        constexpr auto NumGroupsToMerge        = GroupedConvTraitsType_::NumGroupsToMerge;
        // clang-format off
        return concat('_', "grouped_convolution_backward_weight", 
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
        static_assert(ck_tile::reflect::HasInstanceTraits<GroupedConvolutionBackwardWeightKernel>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_tile_grouped_convolution_backward_weight.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<GroupedConvolutionBackwardWeightKernel>();
    }
#endif

    CK_TILE_HOST static constexpr auto
    GridSize(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN), kargs.GemmBatch, kargs.k_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr GroupedConvBwdWeightKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvBwdWeightHostArgs& hostArgs)
    {
        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::cout << "MPerBlock: " << number<TilePartitioner::MPerBlock>{} << std::endl;
            std::cout << "NPerBlock: " << number<TilePartitioner::NPerBlock>{} << std::endl;
            std::cout << "KPerBlock: " << number<TilePartitioner::KPerBlock>{} << std::endl;
        }

        auto kernel_args = GroupedConvBwdWeightKernelArgsSpecialized(hostArgs);

        using KernelImpl = GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                  TilePartitioner_,
                                                                  GemmPipeline_,
                                                                  EpiloguePipeline_>;

        // Negative k_batch value: split-K autodeduction.
        if(kernel_args.k_batch < 0)
        {
            const auto optimal_split_k =
                calculate_optimal_k_batch<GemmPipeline_::BlockSize, KernelImpl, TilePartitioner_>(
                    kernel_args);
            kernel_args.k_batch = optimal_split_k;
        }

        return kernel_args;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        if(kargs.k_batch < 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(
                    "k_batch must be at least one. Ensure argument is created via MakeKernelArgs.");
            }
            return false;
        }

        if constexpr(!std::is_same_v<typename EpiloguePipeline::ODataType, float> &&
                     !std::is_same_v<typename EpiloguePipeline::ODataType, double>)
        {
            // The epilogue performs atomic add related to split-K using the ODataType.
            // If the type is less accurate than float, large split-K values may lead to
            // accuracy issues. Hence, we limit the maximum split-K value to 128 in such cases.
            if(kargs.k_batch > 128)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "For epilogue output data type that is not float/double, we must have "
                        "k_batch <= 128.");
                }
                return false;
            }
        }

        if constexpr((GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                      is_any_of<WeiDataType, fp16_t, bf16_t>::value))
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for K_batch > 1!");
                }
                return false;
            }
        }

        if(kargs.GemmK < TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{}) * kargs.k_batch)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("KBatch is too large, part of GPU wouldn't be utilized!");
            }
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

        if constexpr(GroupedConvTraitsType_::ExplicitGemm &&
                     ConvSpecialization != ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            CK_TILE_ERROR(
                "Explicit Gemm is supported only for Filter1x1Stride1Pad0 specialization!");
            return false;
        }

        namespace ctc = tensor_layout::convolution;

        if constexpr(std::is_same_v<InLayout, ctc::NWGC> || std::is_same_v<InLayout, ctc::NHWGC> ||
                     std::is_same_v<InLayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(ConvC % GroupedConvTraitsType_::VectorSizeB != 0)
            {
                CK_TILE_ERROR("Conv C is not a multiple of vector load size for "
                              "input image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported input layout!");
            return false;
        }

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
                CK_TILE_ERROR("Conv K is not a multiple of vector store size "
                              "for output image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported output layout!");
            return false;
        }

        if constexpr(GroupedConvTraitsType_::NumGroupsToMerge > 1)
        {
            const index_t ConvG = kargs.wei_g_k_c_xs_lengths[number<0>{}];
            if(ConvG % GroupedConvTraitsType_::NumGroupsToMerge != 0)
            {
                CK_TILE_ERROR("ConvG must be a multiple of NumGroupsToMerge!");
                return false;
            }
        }

        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeCBlockWindow(WeiDataType* c_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_m,
                     const index_t block_idx_n)
    {
        const auto& c_tensor_view =
            make_tensor_view<address_space_enum::global, DstInMemOp>(c_ptr, kargs.c_grid_desc_m_n);

        const auto& c_pad_view = pad_tensor_view(
            c_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<true, true>{});

        return make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m, block_idx_n});
    }

    CK_TILE_DEVICE static auto
    MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                      const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                      const index_t block_idx_m,
                      const index_t block_idx_n)
    {
        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                static_assert(std::is_same_v<std::tuple_element_t<i, DsLayout>, OutLayout>,
                              "Not supported!");
                static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                              "Not supported!");
                static_assert(std::is_same_v<std::tuple_element_t<i, DsDataType>, WeiDataType>,
                              "Not supported!");

                return make_tensor_view<address_space_enum::global>(
                    static_cast<WeiDataType*>(ds_ptr[i]), kargs.c_grid_desc_m_n);
            },
            number<NumDTensor>{});

        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(ds_tensor_view[i],
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<true, true>{});
            },
            number<NumDTensor>{});

        return generate_tuple(
            [&](auto i) {
                return make_tile_window(ds_pad_view[i],
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {block_idx_m, block_idx_n});
            },
            number<NumDTensor>{});
    }

    CK_TILE_DEVICE static auto
    MakeBBlockWindow(const InDataType* b_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_n,
                     const index_t block_idx_k)
    {
        static_assert(!GemmPipeline::BlockGemmShape::PermuteB, "Not implemented!");
        const auto& b_tensor_view =
            make_tensor_view<address_space_enum::global>(b_ptr, kargs.b_grid_desc_k_n);

        const auto& b_pad_view =
            pad_tensor_view(b_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock>{} * kargs.k_batch,
                                       number<TilePartitioner::NPerBlock>{}),
                            sequence<true, true>{});

        return make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_k, block_idx_n});
    }

    CK_TILE_DEVICE static auto
    MakeABlockWindow(const OutDataType* a_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_m,
                     const index_t block_idx_k)
    {
        static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view =
            make_tensor_view<address_space_enum::global>(a_ptr, kargs.a_grid_desc_k_m);

        const auto& a_pad_view =
            pad_tensor_view(a_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock>{} * kargs.k_batch,
                                       number<TilePartitioner::MPerBlock>{}),
                            sequence<true, true>{});

        return make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::MPerBlock>{}),
            {block_idx_k, block_idx_m});
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs Grouped Convolution Backward Weight kernel arguments
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const OutDataType* a_ptr,
                                       const InDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       WeiDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                       const index_t num_loop,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const index_t block_idx_k)
    {
        // Create block windows using helper methods
        const auto& a_block_window = MakeABlockWindow(a_ptr, kargs, block_idx_m, block_idx_k);
        const auto& b_block_window = MakeBBlockWindow(b_ptr, kargs, block_idx_n, block_idx_k);
        const auto& d_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);

        // Run Epilogue Pipeline with k_batch dispatching
        if(kargs.k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_0);
        }
        else
        {
            if constexpr(!(GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                           is_any_of<WeiDataType, fp16_t, bf16_t>::value))
            {
                auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                    c_ptr, kargs, block_idx_m, block_idx_n);

                EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_0);
            }
        }
    }

    CK_TILE_DEVICE void CallExplicitGemm(GroupedConvBwdWeightKernelArgsSpecialized& kargs) const
    {
        static_assert(NumDTensor == 0, "Not supported!");
        using ExplicitBatchedGemmKernel =
            BatchedGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto batched_gemm_kargs = typename ExplicitBatchedGemmKernel::BatchedGemmKernelArgs{
            {{kargs.out_ptr},
             {kargs.in_ptr},
             {},
             kargs.wei_ptr,
             kargs.GemmM,
             kargs.GemmN,
             kargs.GemmK,
             {kargs.GemmM * kargs.GemmBatch},
             {kargs.GemmN * kargs.GemmBatch},
             {},
             kargs.GemmN,
             kargs.k_batch},
            kargs.GemmM,
            kargs.GemmN,
            kargs.GemmM * kargs.GemmN,
            kargs.GemmBatch};
        ExplicitBatchedGemmKernel{}(batched_gemm_kargs);
    }

    CK_TILE_DEVICE void operator()(GroupedConvBwdWeightKernelArgsSpecialized& kargs) const
    {
        if constexpr(GroupedConvTraitsType_::ExplicitGemm)
        {
            CallExplicitGemm(kargs);
        }
        else
        {
            const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
            const auto [iM, iN] =
                TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(blockIdX);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const auto blockIdZ    = amd_wave_read_first_lane(blockIdx.z);
            const index_t num_loop = amd_wave_read_first_lane(ck_tile::integer_divide_ceil(
                kargs.GemmK, kargs.k_batch * TilePartitioner::KPerBlock));
            const index_t i_k =
                amd_wave_read_first_lane(blockIdZ * num_loop * TilePartitioner::KPerBlock);

            const auto blockIdY       = amd_wave_read_first_lane(blockIdx.y);
            const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * blockIdY);
            const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * blockIdY);
            const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * blockIdY);

            // options
            // conv_bwd_weight = Out * In = Weight
            const OutDataType* a_ptr =
                static_cast<const OutDataType*>(kargs.out_ptr) + group_offset_a;
            const InDataType* b_ptr = static_cast<const InDataType*>(kargs.in_ptr) + group_offset_b;
            WeiDataType* c_ptr      = static_cast<WeiDataType*>(kargs.wei_ptr) + group_offset_c;

            __shared__ char smem_ptr[GetSmemSize()];

            RunGemm(a_ptr, b_ptr, kargs.ds_ptr, c_ptr, smem_ptr, kargs, num_loop, i_m, i_n, i_k);
        }
    }
};

} // namespace ck_tile
