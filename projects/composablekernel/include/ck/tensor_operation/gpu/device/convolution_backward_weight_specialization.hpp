// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionBackwardWeightSpecialization
{
    Default,
    Filter1x1Stride1Pad0,
    Filter1x1Pad0,
    OddC,
};

inline std::string
getConvBackwardWeightSpecializationString(const ConvolutionBackwardWeightSpecialization& s)
{
    switch(s)
    {
    case ConvolutionBackwardWeightSpecialization::Default: return "Default";
    case ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0:
        return "Filter1x1Stride1Pad0";
    case ConvolutionBackwardWeightSpecialization::Filter1x1Pad0: return "Filter1x1Pad0";
    case ConvolutionBackwardWeightSpecialization::OddC: return "OddC";
    default: return "Unrecognized specialization!";
    }
}
} // namespace device
} // namespace tensor_operation
} // namespace ck
