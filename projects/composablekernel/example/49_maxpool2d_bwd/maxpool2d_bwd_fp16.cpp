// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "maxpool2d_bwd_common.hpp"

using InDataType      = ck::half_t;
using OutDataType     = ck::half_t;
using IndexDataType   = int32_t;
using ComputeDataType = float;
using DInDataType     = ck::half_t;
using DOutDataType    = ck::half_t;

static constexpr bool PropagateNan = false;

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    // Pool shape
    ck::index_t N                 = 1;
    ck::index_t C                 = 1;
    ck::index_t Y                 = 3;
    ck::index_t X                 = 3;
    ck::index_t Hi                = 32;
    ck::index_t Wi                = 32;
    ck::index_t window_stride_h   = 1;
    ck::index_t window_stride_w   = 1;
    ck::index_t window_dilation_h = 1;
    ck::index_t window_dilation_w = 1;
    ck::index_t in_left_pad_h     = 0;
    ck::index_t in_left_pad_w     = 0;
    ck::index_t in_right_pad_h    = 0;
    ck::index_t in_right_pad_w    = 0;

    bool pass = maxpool_bwd_test<InDataType,
                                 OutDataType,
                                 IndexDataType,
                                 ComputeDataType,
                                 DInDataType,
                                 DOutDataType,
                                 PropagateNan>(do_verification,
                                               time_kernel,
                                               N,
                                               C,
                                               Y,
                                               X,
                                               Hi,
                                               Wi,
                                               window_stride_h,
                                               window_stride_w,
                                               window_dilation_h,
                                               window_dilation_w,
                                               in_left_pad_h,
                                               in_left_pad_w,
                                               in_right_pad_h,
                                               in_right_pad_w);

    return (pass ? 0 : 1);
}
