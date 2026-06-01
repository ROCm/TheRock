// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_avgpool3d_bwd_ndhwc_ndhwc.hpp"

#include "avgpool3d_bwd_common.hpp"

using DOutDataType    = ck::half_t;
using DInDataType     = ck::half_t;
using ComputeDataType = float;

#if 1
using DOutLayout = ck::tensor_layout::convolution::NDHWC;
using DInLayout  = ck::tensor_layout::convolution::NDHWC;
#else
using DOutLayout = ck::tensor_layout::convolution::NCDHW;
using DInLayout  = ck::tensor_layout::convolution::NCDHW;
#endif

using DevicePoolBwdInstance =
    ck::tensor_operation::device::DeviceAvgPool3dBwd_NDHWC_NDHWC<DOutDataType,
                                                                 DInDataType,
                                                                 ComputeDataType,
                                                                 64, // BlockSize
                                                                 64, // ReduceMThreadClusterSize
                                                                 1,  // ReduceKThreadClusterSize
                                                                 1,  // ReduceMThreadSliceSize
                                                                 1,  // ReduceKThreadSliceSize
                                                                 1>; // InSrcOutDstVectorSize

int main()
{
    std::vector<ck::index_t> window_lengths    = {5, 5, 5};
    std::vector<ck::index_t> window_strides    = {2, 2, 2};
    std::vector<ck::index_t> window_dilations  = {2, 2, 2};
    std::vector<ck::index_t> dinput_left_pads  = {0, 0, 0};
    std::vector<ck::index_t> dinput_right_pads = {0, 0, 0};

    ck::index_t N  = 1;
    ck::index_t C  = 16;
    ck::index_t Di = 40;
    ck::index_t Hi = 40;
    ck::index_t Wi = 40;

    pool3d_bwd_test<DevicePoolBwdInstance, DOutDataType, DInDataType, DOutLayout, DInLayout>(
        true,
        false,
        N,
        C,
        Di,
        Hi,
        Wi,
        window_lengths,
        window_strides,
        window_dilations,
        dinput_left_pads,
        dinput_right_pads);
}
