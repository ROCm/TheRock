// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::GNWC;
using WeiLayout = ck::tensor_layout::convolution::GKXC;
using OutLayout = ck::tensor_layout::convolution::GNWK;

static constexpr ck::index_t NumDimSpatial = 1;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Wo            = 28;
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_lengths{G, N, C, Wi};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> filter_lengths{G, K, C, X};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_lengths{G, N, K, Wo};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_strides{N * Wi * C, Wi* C, 1, C};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> weights_strides{K * X * C, X* C, 1, C};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_strides{N * Wo * K, Wo* K, 1, K};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_left_pads{1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_right_pads{1};

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
