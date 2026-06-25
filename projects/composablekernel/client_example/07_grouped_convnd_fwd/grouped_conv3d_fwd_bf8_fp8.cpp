// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::bf8_t;
using WeiDataType = ck::f8_t;
using OutDataType = ck::f8_t;

using InLayout  = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWGK;

using AComputeType = ck::bf8_t;
using BComputeType = ck::f8_t;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 1;
static constexpr ck::index_t N             = 64;
static constexpr ck::index_t K             = 128;
static constexpr ck::index_t C             = 64;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 28;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 3;
static constexpr ck::index_t Do            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 3;

int main()
{
    return run_grouped_conv_fwd<NumDimSpatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InLayout,
                                WeiLayout,
                                OutLayout,
                                3,
                                AComputeType,
                                BComputeType>(
               {N, Di, Hi, Wi, G, C}, {G, K, Z, Y, X, C}, {N, Do, Ho, Wo, G, K})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
