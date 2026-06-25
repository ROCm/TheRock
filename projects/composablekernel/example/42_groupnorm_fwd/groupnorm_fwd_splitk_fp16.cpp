// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

using XDataType              = ck::half_t;
using GammaDataType          = ck::half_t;
using BetaDataType           = ck::half_t;
using YDataType              = ck::half_t;
using SaveMeanInvStdDataType = float;
using ComputeDataType        = float;
using YElementOp             = ck::tensor_operation::element_wise::Swish;

#define SAVE_MEAN_INV_STD

using DeviceInstance = ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<
    XDataType,
    GammaDataType,
    BetaDataType,
    ComputeDataType,
    YDataType,
    SaveMeanInvStdDataType,
    YElementOp,
    Rank,
    NumReduceDim,
    256, // BlockSize
    1,   // ClusterM
    256, // ClusterK
    1,   // SliceM
    16,  // SliceK
    1,   // SrcVecDim (0=M, 1=K)
    2,   // SrcScalarPerVector
    1,   // GammaVecDim (0=M, 1=K)
    2,   // GammaScalarPerVector
    1,   // BetaVecDim (0=M, 1=K)
    2,   // BetaScalarPerVector
    2,   // YScalarPerVector
    1>;  // SaveMeanInvStdScalarPerVector

#include "run_groupnorm_fwd_example.inc"

int main(int argc, char* argv[]) { run_groupnorm_fwd_example(argc, argv); }
