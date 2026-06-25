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

#define SAVE_MEAN_INV_STD

struct YElementOp
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(ck::is_same<X, float>::value || ck::is_same<X, double>::value ||
                          ck::is_same<X, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        static_assert(ck::is_same<Y, float>::value || ck::is_same<Y, double>::value ||
                          ck::is_same<Y, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        X a;

        ck::tensor_operation::element_wise::Sigmoid{}(a, x);

        y = ck::type_convert<Y>(x * a);
    };
};

using DeviceInstance =
    ck::tensor_operation::device::DeviceNormalizationFwdImpl<XDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             ComputeDataType,
                                                             YDataType,
                                                             SaveMeanInvStdDataType,
                                                             YElementOp,
                                                             Rank,
                                                             NumReduceDim,
                                                             1024, // BlockSize
                                                             1,    // ClusterM
                                                             1024, // ClusterK
                                                             1,    // SliceM
                                                             32,   // SliceK
                                                             1,    // SrcVecDim (0=M, 1=K)
                                                             2,    // SrcScalarPerVector
                                                             1,    // GammaVecDim (0=M, 1=K)
                                                             2,    // GammaScalarPerVector
                                                             1,    // BetaVecDim (0=M, 1=K)
                                                             2,    // BetaScalarPerVector
                                                             2,    // YScalarPerVector
                                                             1>;   // SaveMeanInvStdScalarPerVector

#include "run_groupnorm_fwd_example.inc"

int main(int argc, char* argv[]) { run_groupnorm_fwd_example(argc, argv); }
