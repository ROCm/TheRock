// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout    = ck::tensor_layout::convolution::GNWC;
using WeiLayout   = ck::tensor_layout::convolution::GKXC;
using OutLayout   = ck::tensor_layout::convolution::GNWK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr ck::index_t NumDimSpatial = 1;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Wo            = 28;

int main()
{
    return run_grouped_conv_fwd<NumDimSpatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InLayout,
                                WeiLayout,
                                OutLayout,
                                3>({N, Wi, G, C}, {G, K, X, C}, {N, Wo, G, K})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
