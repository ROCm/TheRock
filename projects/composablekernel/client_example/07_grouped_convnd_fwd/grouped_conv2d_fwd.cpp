// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout    = ck::tensor_layout::convolution::NHWGC;
using WeiLayout   = ck::tensor_layout::convolution::GKYXC;
using OutLayout   = ck::tensor_layout::convolution::NHWGK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256; // batch size
static constexpr ck::index_t K             = 64;  // output channel
static constexpr ck::index_t C             = 32;  // input channel (per group)
static constexpr ck::index_t Y             = 3;   // filter H
static constexpr ck::index_t X             = 3;   // filter W
static constexpr ck::index_t Hi            = 28;  // input H
static constexpr ck::index_t Wi            = 28;  // input W
static constexpr ck::index_t Ho            = 28;  // output H
static constexpr ck::index_t Wo            = 28;  // output W

int main()
{
    return run_grouped_conv_fwd<NumDimSpatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InLayout,
                                WeiLayout,
                                OutLayout,
                                3>({N, Hi, Wi, G, C}, {G, K, Y, X, C}, {N, Ho, Wo, G, K})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
