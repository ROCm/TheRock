// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include <fstream>
#include <variant>

// functions to return the corresponding structs based on generated template parameters

using layouts = std::variant<ck::tensor_layout::convolution::GNWK,
                             ck::tensor_layout::convolution::GNHWK,
                             ck::tensor_layout::convolution::NHWGK,
                             ck::tensor_layout::convolution::GNDHWK,
                             ck::tensor_layout::convolution::NDHWGK>;
// return the layout type: currently this is the only type supported in MIOpen
auto layout_type(std::string type)
{
    if(type == "ck::tensor_layout::convolution::NHWGK")
    {
        return ck::tensor_layout::convolution::NHWGK{};
    }
    throw std::runtime_error("Incorrect layout");
}
// return the right gemm spec based on the generated template parameters
ck::tensor_operation::device::GemmSpecialization gemm_type(std::string type)
{
    if(type == "ck::tensor_operation::device::GemmSpecialization::Default")
    {
        return ck::tensor_operation::device::GemmSpecialization::Default;
    }
    if(type == "ck::tensor_operation::device::GemmSpecialization::MNKPadding")
    {
        return ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    }
    throw std::runtime_error("Incorrect gemm spec: " + type);
}

// return the type of convolution
ck::tensor_operation::device::ConvolutionForwardSpecialization conv_type(std::string type)
{
    if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    }
    if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
    }
    if(type ==
       "ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;
    }
    if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;
    }
    throw std::runtime_error("Incorrect conv spec: " + type);
}

// Function to call on MatrixPadder via a wrapper struct
// NOTE: CK only uses MNKPadding for forward convolution
template <typename CDesc_MRaw_NRaw>
auto pad(ck::index_t mpb,
         ck::index_t npb,
         ck::index_t kpb,
         ck::tensor_operation::device::GemmSpecialization gemm,
         CDesc_MRaw_NRaw conv)
{
    if(gemm == ck::tensor_operation::device::GemmSpecialization::MNKPadding)
    {
        ck::tensor_operation::device::MatrixPadder<
            ck::tensor_operation::device::GemmSpecialization::MNKPadding,
            ck::index_t,
            ck::index_t,
            ck::index_t>
            a;
        a.MPerTile_ = mpb;
        a.NPerTile_ = npb;
        a.KPerTile_ = kpb;
        auto tmp    = grid_desc(a, conv);
        return tmp;
    }
    throw std::runtime_error("Incorrect template parameters, check gemm spec");
}

// Functions to call on TransformConvFwdToGemm through wrapper: different functions based on num
// dims
// FIXME: add a way to properly pass in the layout
auto transform_conv(ck::index_t num_dim,
                    ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                    ck::Array<ck::index_t, 5> out_lengths,
                    ck::Array<ck::index_t, 5> out_strides)
{
    ck::Array<ck::index_t, 5> dummy_dims;
    ck::Array<ck::index_t, 2> dummy_spatial_dims;
    if(num_dim == 2 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 2 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 2 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 2 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    throw std::runtime_error("Incorrect conv spec");
}

auto transform_conv_3d(ck::index_t num_dim,
                       ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                       ck::Array<ck::index_t, 6> out_lengths,
                       ck::Array<ck::index_t, 6> out_strides)
{
    ck::Array<ck::index_t, 6> dummy_dims;
    ck::Array<ck::index_t, 3> dummy_spatial_dims;

    if(num_dim == 3 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 3 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 3 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 3 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            3,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    throw std::runtime_error("Incorrect conv spec");
}

auto transform_conv_1d(ck::index_t num_dim,
                       ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                       ck::Array<ck::index_t, 4> out_lengths,
                       ck::Array<ck::index_t, 4> out_strides)
{
    ck::Array<ck::index_t, 4> dummy_dims;
    ck::Array<ck::index_t, 1> dummy_spatial_dims;

    if(num_dim == 1 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 1 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 1 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    if(num_dim == 1 && spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            1,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>
            conv_fwd{dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     dummy_dims,
                     out_lengths,
                     out_strides,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims,
                     dummy_spatial_dims};

        auto res = ck::tensor_operation::TransformConv();
        return res.transform_func(conv_fwd);
    }
    throw std::runtime_error("Incorrect dims or conv spec");
}

template <typename CGridDesc_M_N>
auto block_2_etile(ck::index_t m_per_block, ck::index_t n_per_block, CGridDesc_M_N matrix_padder)
{
    if(m_per_block == 32 && n_per_block == 64)
    {
        auto b2e = ck::BlockToCTileMap_M00_N0_M01Adapt<32, 64, CGridDesc_M_N>(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 32 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<32, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 64 && n_per_block == 32)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 32, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 64 && n_per_block == 64)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 64, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 64 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<64, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 128 && n_per_block == 32)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 32, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 128 && n_per_block == 64)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 64, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 128 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 128 && n_per_block == 256)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 256, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    if(m_per_block == 256 && n_per_block == 128)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<256, 128, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    throw std::runtime_error("Incorrect template parameters");
}

// wrapper functions by dims to get grid size - uses above 3 functions
// TODO: eventually remove the 1d/2d versions as CK will only support 3d convolutions
auto get_launch_params_1d(ck::host::Solution solution,
                          ck::Array<ck::index_t, 4> out_lengths,
                          ck::Array<ck::index_t, 4> out_strides)
{
    auto num_dim     = solution.GetTemplateParameter<ck::index_t>("NumDim");
    auto m_per_block = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
    auto n_per_block = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
    auto k_per_block = solution.GetTemplateParameter<ck::index_t>("KPerBlock");
    auto GemmType    = solution.GetTemplateParameter<std::string>("GemmSpecialization");
    auto ConvType    = solution.GetTemplateParameter<std::string>("ConvSpecialization");
    ck::tensor_operation::device::GemmSpecialization GemmSpec               = gemm_type(GemmType);
    ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec = conv_type(ConvType);
    auto conv_to_gemm_transformer = transform_conv_1d(num_dim, ConvSpec, out_lengths, out_strides);
    auto matrix_padder =
        pad(m_per_block, n_per_block, k_per_block, GemmSpec, conv_to_gemm_transformer);
    auto b2e = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}

auto get_launch_params(ck::host::Solution solution,
                       ck::Array<ck::index_t, 5> out_lengths,
                       ck::Array<ck::index_t, 5> out_strides)
{
    auto num_dim     = solution.GetTemplateParameter<ck::index_t>("NumDim");
    auto m_per_block = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
    auto n_per_block = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
    auto k_per_block = solution.GetTemplateParameter<ck::index_t>("KPerBlock");
    auto GemmType    = solution.GetTemplateParameter<std::string>("GemmSpecialization");
    auto ConvType    = solution.GetTemplateParameter<std::string>("ConvSpecialization");
    ck::tensor_operation::device::GemmSpecialization GemmSpec               = gemm_type(GemmType);
    ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec = conv_type(ConvType);
    auto conv_to_gemm_transformer = transform_conv(num_dim, ConvSpec, out_lengths, out_strides);
    auto matrix_padder =
        pad(m_per_block, n_per_block, k_per_block, GemmSpec, conv_to_gemm_transformer);
    auto b2e = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}

auto get_launch_params_3d(ck::host::Solution solution,
                          ck::Array<ck::index_t, 6> out_lengths,
                          ck::Array<ck::index_t, 6> out_strides)
{
    auto num_dim     = solution.GetTemplateParameter<ck::index_t>("NumDim");
    auto m_per_block = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
    auto n_per_block = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
    auto k_per_block = solution.GetTemplateParameter<ck::index_t>("KPerBlock");
    auto GemmType    = solution.GetTemplateParameter<std::string>("GemmSpecialization");
    auto ConvType    = solution.GetTemplateParameter<std::string>("ConvSpecialization");
    ck::tensor_operation::device::GemmSpecialization GemmSpec               = gemm_type(GemmType);
    ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec = conv_type(ConvType);
    auto conv_to_gemm_transformer = transform_conv_3d(num_dim, ConvSpec, out_lengths, out_strides);
    auto matrix_padder =
        pad(m_per_block, n_per_block, k_per_block, GemmSpec, conv_to_gemm_transformer);
    auto b2e = block_2_etile(m_per_block, n_per_block, matrix_padder);
    return b2e;
}
