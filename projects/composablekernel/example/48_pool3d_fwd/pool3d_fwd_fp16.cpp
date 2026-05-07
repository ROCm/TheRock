// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "pool3d_fwd_common.hpp"

using InDataType      = ck::half_t;
using OutDataType     = ck::half_t;
using ComputeDataType = float;

using IndexDataType = int32_t;

using InLayout  = ck::tensor_layout::convolution::NDHWC;
using OutLayout = ck::tensor_layout::convolution::NDHWC;

#if 1
static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;
#else
static constexpr auto ReduceOpId = ck::ReduceTensorOp::AVG;
#endif

static constexpr bool OutputIndex  = false;
static constexpr bool PropagateNan = false;

using DevicePoolFwdInstance =
    ck::tensor_operation::device::DevicePool3dFwd_NDHWC_NDHWC<InDataType,
                                                              OutDataType,
                                                              IndexDataType,
                                                              ComputeDataType,
                                                              ReduceOpId,
                                                              OutputIndex,
                                                              64, // BlockSize
                                                              64, // ReduceMThreadClusterSize
                                                              1,  // ReduceKThreadClusterSize
                                                              1,  // ReduceMThreadSliceSize
                                                              1,  // ReduceKThreadSliceSize
                                                              1>; // InSrcOutDstVectorSize

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    // Pool shape
    ck::index_t N                 = 2;
    ck::index_t C                 = 32;
    ck::index_t Z                 = 2;
    ck::index_t Y                 = 2;
    ck::index_t X                 = 2;
    ck::index_t Di                = 30;
    ck::index_t Hi                = 30;
    ck::index_t Wi                = 30;
    ck::index_t window_stride_d   = 2;
    ck::index_t window_stride_h   = 2;
    ck::index_t window_stride_w   = 2;
    ck::index_t window_dilation_d = 1;
    ck::index_t window_dilation_h = 1;
    ck::index_t window_dilation_w = 1;
    ck::index_t in_left_pad_d     = 1;
    ck::index_t in_left_pad_h     = 1;
    ck::index_t in_left_pad_w     = 1;
    ck::index_t in_right_pad_d    = 1;
    ck::index_t in_right_pad_h    = 1;
    ck::index_t in_right_pad_w    = 1;

    bool pass = pool3d_test<DevicePoolFwdInstance,
                            InDataType,
                            OutDataType,
                            ComputeDataType,
                            IndexDataType,
                            InLayout,
                            OutLayout,
                            ReduceOpId,
                            PropagateNan,
                            OutputIndex>(do_verification,
                                         time_kernel,
                                         N,
                                         C,
                                         Z,
                                         Y,
                                         X,
                                         Di,
                                         Hi,
                                         Wi,
                                         window_stride_d,
                                         window_stride_h,
                                         window_stride_w,
                                         window_dilation_d,
                                         window_dilation_h,
                                         window_dilation_w,
                                         in_left_pad_d,
                                         in_left_pad_h,
                                         in_left_pad_w,
                                         in_right_pad_d,
                                         in_right_pad_h,
                                         in_right_pad_w);

    return (pass ? 0 : 1);
}
