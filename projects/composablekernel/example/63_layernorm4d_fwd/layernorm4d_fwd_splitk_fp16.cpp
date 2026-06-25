// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using XDataType              = ck::half_t;
using GammaDataType          = ck::half_t;
using BetaDataType           = ck::half_t;
using YDataType              = ck::half_t;
using SaveMeanInvStdDataType = float;
using ComputeDataType        = float;
using PassThrough            = ck::tensor_operation::element_wise::PassThrough;

#define SAVE_MEAN_INV_STD

constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

using DeviceInstance = ck::tensor_operation::device::DeviceNormalizationFwdSplitKImpl<
    XDataType,
    GammaDataType,
    BetaDataType,
    ComputeDataType,
    YDataType,
    SaveMeanInvStdDataType,
    PassThrough,
    Rank,
    NumReduceDim,
    256, // BlockSize
    8,   // ClusterM
    32,  // ClusterK
    1,   // SliceM
    8,   // SliceK
    1,   // XYVectorDim (0=M, 1=K)
    8,   // XScalarPerVector
    1,   // GammaVecDim (0=M, 1=K)
    8,   // GammaScalarPerVector
    1,   // BetaVecDim (0=M, 1=K)
    8,   // BetaScalarPerVector
    8,   // YScalarPerVector
    1>;  // SaveMeanInvStdScalarPerVector

#include "run_layernorm4d_fwd_example.inc"

int main() { return run_layernorm4d_fwd_example<DeviceInstance>(); }
