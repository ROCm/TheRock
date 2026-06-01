// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::GNDHWC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::GNDHWK;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 8;
static constexpr ck::index_t N             = 64;
static constexpr ck::index_t K             = 128;
static constexpr ck::index_t C             = 128;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 28;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 3;
static constexpr ck::index_t Do            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 3;
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_lengths{G, N, C, Di, Hi, Wi};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> filter_lengths{G, K, C, Z, Y, X};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_lengths{G, N, K, Do, Ho, Wo};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_strides{
    N * Di * Hi * Wi * C, Di* Hi* Wi* C, 1, Hi* Wi* C, Wi* C, C};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> weights_strides{
    K * Z * Y * X * C, Z* Y* X* C, 1, Y* X* C, X* C, C};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_strides{
    N * Do * Ho * Wo * K, Do* Ho* Wo* K, 1, Ho* Wo* K, Wo* K, K};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1, 1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1, 1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1, 1};

int main()
{
    return run_grouped_conv_bwd_weight<NumDimSpatial,
                                       InDataType,
                                       WeiDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(input_lengths,
                                                  input_strides,
                                                  filter_lengths,
                                                  weights_strides,
                                                  output_lengths,
                                                  output_strides,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
